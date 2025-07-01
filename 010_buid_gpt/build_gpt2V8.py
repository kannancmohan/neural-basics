""" 
Prerequisite: Prior to running this code, please download the training dataset using 'python fineweb.py' ).

run the code using command 'torchrun --standard --nproc_per_node=2 build_gpt2V8.py'

"""

import sys
import torch
import numpy as np
import math
import tiktoken
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import time
import inspect
import os
from hellaswag import render_hellaswag_example, iterate_hellaswag_examples, get_most_likely_row
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist # package for distributed training - that provides the low-level tools for multi-GPU and multi-node coordination

##--------MODEL-------- ##
@dataclass
class GPTConfig: # this matches the original gpt hyper parameters
    block_size: int = 1024 # context_length/max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # The number of transformer layers.
    n_head: int = 12 # The number of attention heads in each multi-head attention module.
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # the transformer module
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),# input token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),# position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #list of transformer blocks. 12 blocks in this case
            ln_f = nn.LayerNorm(config.n_embd), # final layerNorm as per transformer architecture
        ))
        # the language-model-head(lm_head) module - the classifier part
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # tied weights

        # weight and bias initilization scheme
        self.apply(self._init_weights) # here the self.apply recursively apply _init_weights on every sub module and self

    def _init_weights(self, module):# initilization of weight and bias similar to gpt2
        if isinstance(module, nn.Linear):
            std = 0.02
            # handling special case of initializing weights in layers that are part of a residual connection
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # adjusting std for feedforward and attention layer to fix std growth inside residula stream
                # 2 here, because there are two residual connections within each transformer block
                std *= (2 * self.config.n_layer) ** -0.5 # its same as 1 / sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # creates a tensor containing values from 0 to T-1 - shape (T)
        pos_emb = self.transformer.wpe(pos) # Looks up the positional embeddings for the positions - shape (T, n_embd). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
        # pos_emb = self.transformer.wpe(pos).clone() # Looks up the positional embeddings for the positions - shape (T, n_embd). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
        tok_emb = self.transformer.wte(idx) # Looks up the token embeddings - shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        ## construct a GPTConfig with the gpt2 hyperparameters ##
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)

        ## Init the custom model and capture its learned params##
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # capture all learned parameters except the mask/buffer param(they are not learned param)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        ## Init pretrained huggingface/transformers model to gets its weights ##
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        ## copy over the parameter values from pre-trained model to our custom models paramete"
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # transposing ?
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    #--- weight decay method ---#
    def configure_optimizers(self, weight_decay, learning_rate, device):
        """  Find and apply decay only on params that are weight decayable """

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # weight decay on 2D param
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # dimension less than 2. eg biases, layernorms
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the (kernel)fused version(i.e to fuse all param weight-decay updates) if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print(f"using fused AdamW optimizer: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        # the communcation
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        # the computation
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # feed forward

    def forward(self, x):
        # here is the skip/residual connection - where we add the input to the output of the layer
        # also, we are invoking the LayerNorm layers(self.ln1(x) and self.ln2(x) ) before the skip/residual calculation
        x = x + self.attn(self.ln_1(x)) # first skip/residual connection
        x = x + self.mlp(self.ln_2(x)) # second skip/residual connection
        return x

class MLP(nn.Module):

    """ FeedForward inside Block: a simple linear layer followed by a non-linearity and then by another linear layer """

    def __init__(self, config):
        super().__init__()
        # Standard practice of multiplying by 4 -  a common technique to give the FeedForward network more capacity to process the contextually enriched features
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # The first linear layer that expands the dimensionality.
        self.gelu    = nn.GELU(approximate='tanh') # The GELU activation function
        # The second linear layer that projects the dimensionality back to the embedding size
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)# added for skip/residual connection - allows the model to learn the best way to linearly combine these non-linear features
        self.c_proj.NANOGPT_SCALE_INIT = 1 # added for fixing growth of standarddeivation in residual connect

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    """ multiple heads of self-attention in parallel - attention mechanism with a causal mask """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #A linear layer that computes the query, key, and value projections for all heads
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)# A linear layer that projects the concatenated outputs of the attention heads back to the embedding size.
        self.c_proj.NANOGPT_SCALE_INIT = 1 # added for fixing growth of standarddeivation in residual connect
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # creates a lower triangular matrix of ones for masking - register it as buffer as they are not trainable parameters
        # self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x) # forward through query,key and value
        q, k, v = qkv.split(self.n_embd, dim=2) # split (B, T, 3 * C) to three (B, T, C) each
        # converting k,q,v to batches of nh(no of heads) so as to do calculations in parallel later
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # compute attention scores ("affinities")
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # auto aggressive mask - which make suer that the tokens attends only to token before them
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # causal mask
        # att = F.softmax(att, dim=-1) # normalizes so that attention sums to 1
        # y = att @ v # perform the weighted aggregation of the values

        ## Flash attention - replacement for the above 4 line of code##
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # concat the results of all head
        # here transpose(1, 2) swaps the dimensions . ie from (B, nh, T, hs) to (B, T, nh, hs)
        # contiguous - This method ensures that the tensor is stored contiguously in memory. Transposing a tensor can sometimes result in a non-contiguous memory layout, which can cause issues or performance penalties with subsequent operations like view. Calling contiguous() creates a new tensor with the same data but a contiguous memory arrangement if it wasn't already contiguous.
        # .view(B, T, C) - This reshapes the tensor back into the original embedding dimension shape. The values from the nh heads, each with hs dimensions, are laid out side-by-side to form a single vector of size C
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
##--------Dataloader-------- ##
# Dataloader implementation
# In Distributed Data Parallel(DDP) training, this DataLoader ensures that each GPU
# processes a unique and non-overlapping portion of the dataset for each training step.
# It achieves this by offsetting the starting position for each process based on its rank
# and by advancing the position by the total batch size across all processes after each batch.

def load_tokens(filename):
    """ load the shareded .npy file  """
    npt = np.load(filename) # loads the NumPy array of tokens from file(.npy)
    # converts the token data to np.int32 and then to a tensor
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, ddp_process_rank, ddp_num_processes, split):
        self.B = B
        self.T = T
        self.ddp_process_rank = ddp_process_rank
        self.ddp_num_processes = ddp_num_processes
        assert split in {'train', 'val'}

        # find all the shard filenames in data_root
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # Offset the starting position based on ddp_process_rank to ensure each process reads a unique portion
        self.current_position = self.B * self.T * self.ddp_process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # construct input(x) and labels(y)
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor by the total batch size across all processes
        # this ensures each process reads a unique, non-overlapping portion of the data in each step
        self.current_position += B * T * self.ddp_num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.ddp_num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.ddp_process_rank # Reset position in the new shard, again offset by rank
        return x, y
   
##--------Helper-------- ##
def get_cuda_details():
    print("Python version:",sys.version)
    print("PyTorch version:",torch.__version__)

    if torch.cuda.is_available():# print cuda details if available
        print("CUDA version used by PyTorch:", torch.version.cuda)
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        cc = torch.cuda.get_device_capability()
        print(f"Compute Capability: {cc[0]}.{cc[1]}")
        print("TF32 supported by GPU:", torch.cuda.get_device_capability()[0] >= 8)
        print("BF16 support in CUDA:", torch.cuda.is_bf16_supported())

##--------TRAIN THE MODEL()-------- ##
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
get_cuda_details()

invoke_eval = True
invoke_hellaswag = True
invoke_sampling = True
use_compile = True
save_checkpoint = True

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

####-- Hyper parameters --####
# ensure that (batch_size * ctx_len * ddp_world_size * grad_accum_steps) = total_batch_size
# setting total_batch_size per training step to ~0.5M, so as to comply with gpt2 batchsize. Here we have used a nice batch size that is power of two
total_batch_size = 524288 # desire batch size per step of the training
batch_size = 16 # the micro batch size - no of independent sequences processed by a single GPU
ctx_len = 1024 # the context length - no of tokens in each sequence.
num_epoch = 1
# max_steps calculated based on total_input_token_size(here its 10BT) divided by total tokens in each batch(here its the total_batch_size:524288)
max_steps = 190703 * num_epoch # total steps in training . calculated 10e9/524288 =

####-- learning rate: cosine decay with warmup --####
max_lr = 6e-4 # based on gpt2 paper
min_lr = max_lr * 0.1 # 10% of max_lr. added based on gpt2 paper
warmup_steps = 715 # as per gpt2 paper the do warmup over the first 375 million token. 375e6/524288=715

# equivalent huggingface method  is transformers.get_cosine_schedule_with_warmup
def get_lr(it, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

####-- Distributed Data Parallel setup --####
# after executing 'torchrun' command, the env variables RANK, LOCAL_RANK, and WORLD_SIZE get set
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp: #  use DDP if available, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "need CUDA for DDP" # TODO check whether ddp works for non cuda
    init_process_group(backend='nccl') # init the ddp. setting 'nccl' so as to use cuda
    ddp_rank = int(os.environ['RANK']) # get global rank of current process - i.e the unique id assinged to process across all participating machines/nodes
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # get local rank of current process - i.e the unique id assinged to each process on a single machine/node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # get total no of processes/gpus in the distribution
    device = f'cuda:{ddp_local_rank}' # Sets the device for the current process to the GPU corresponding to its local rank
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else: # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    print(f"using device: {device}")

####-- train,eval,sample the model --####
assert total_batch_size % (batch_size * ctx_len * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (batch_size * ctx_len * ddp_world_size) # no of steps(here its 32) to reached the desired batch size(524288)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)

# using TF32 for better performance
torch.set_float32_matmul_precision('high') # switches to tensore core for the matrix mul involved in the model

# init custom model
model = GPT(GPTConfig(vocab_size=50304)) # Power-of-Two optimization
model.to(device)
if use_compile:
    # model = torch.compile(model, mode="default") # TODO
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) #ddp_local_rank tells DDP which specific GPU the model replica for this process should reside on
raw_model = model.module if ddp else model

# do the eval,sampling,training
train_loader = DataLoaderLite(batch_size, ctx_len, ddp_process_rank=ddp_rank, ddp_num_processes=ddp_world_size, split='train')
if invoke_eval:
    val_loader = DataLoaderLite(batch_size, ctx_len, ddp_process_rank=ddp_rank, ddp_num_processes=ddp_world_size, split='val')

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time() # to capture time taken
    last_step = (step == max_steps - 1)

    #---------------- 1. evaluate the model after every 250 steps  ---------#
    if invoke_eval and (step % 250 == 0 or last_step):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            #---1.1 save the model after every 5000 steps or at last step ----#
            if save_checkpoint and step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # Note: If you want resume training feature, you might also want to add optimizer.state_dict(), rng seeds etc.
                torch.save(checkpoint, checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")

    #---------------- 2. do hellaswag evaulation after every 250 steps --------#
    if invoke_hellaswag and (step % 250 == 0 or last_step): # and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_hellaswag_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank: #condition  ensures that each HellaSwag example is processed by only one of the participating DDP processes
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_hellaswag_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    #---------------- 3. sample from model after every 250 steps  ---------#
    if invoke_sampling and ((step > 0 and step % 250 == 0) or last_step): # and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device) # separate generator to ensure sampling randomness is independent of other random operations in the training loop
        sample_rng.manual_seed(42 + ddp_rank) #this ensures that each gpu process generates a unique set of samples
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    #---------------- 4. train the model  ------------#
    model.train()
    optimizer.zero_grad() # reset the gradients
    loss_accum = 0.0
    # --- gradient accumulation ---- #
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # we want the ddp to do the gradient sync accross gpus only at the last step of gradient accumulation
        # This is to reduced communication overhead and improved training speed
        if ddp:
            model.required_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # Enables autocasting(i.e to reducing datatype to BFLOAT16) for the forward pass (model + loss) .
        # check here what all gets converted https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # average the loss over the mini-batches - done similar to torch.nn.functional.cross_entropy that computes the mean loss over the elements in batch
        loss_accum += loss.detach()
        loss.backward()
        # --- gradient accumulation ends --- #
        # calculate the average loss across all the gpu's
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        #gradient clipping - clip the gradients to prevent exploding gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # cosine decay with warmup
        lr = get_lr(step, max_steps)
        for param_group in optimizer.param_groups: # pytorch way of setting the learning rate in the optimizer
            param_group['lr'] = lr

        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize() # wait until gpu is done
        t1 = time.time()
        dt = t1-t0 #calculate elasped time in sec
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size #how many tokens are we training in one step
        tokens_per_sec = tokens_processed / dt #how many token data are we training
        if master_process:
            # Calculate remaining time
            remaining_steps = max_steps - (step + 1)
            remaining_time_seconds = remaining_steps * dt
            remaining_time_hours = remaining_time_seconds / 3600

            print(f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm:{norm:.4f} | dt:{dt*1000:.2f}ms | tok/sec:{tokens_per_sec:.2f} | remaining time: {remaining_time_hours:.2f} hours")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp: # clean up the distributed environment at the end of DDP training
    destroy_process_group()