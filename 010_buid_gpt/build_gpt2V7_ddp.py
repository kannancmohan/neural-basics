# Improvements 
# update to use Distributed Data Parallel


import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import tiktoken
from torch.nn import functional as F

import time
import inspect

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist # package for distributed training - that provides the low-level tools for multi-GPU and multi-node coordination
import os

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

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # creates a tensor containing values from 0 to T-1 - shape (T)
        pos_emb = self.transformer.wpe(pos) # Looks up the positional embeddings for the positions - shape (T, n_embd)
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the (kernel)fused version(i.e to fuse all param weight-decay updates) if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW optimizer: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        # the communication
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
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

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

#---------------- learning rate: cosine decay with warmup ------------#
max_lr = 6e-4 # based on gpt2 paper
min_lr = max_lr * 0.1 # 10% of max_lr. added based on gpt2 paper
warmup_steps = 10

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

#---------------- DataLoader to load input and labels ------------#
class DataLoaderLite:

    def __init__(self, B, T, ddp_process_rank, ddp_num_processes):
        self.B = B
        self.T = T
        self.ddp_process_rank = ddp_process_rank
        self.ddp_num_processes = ddp_num_processes
        # load the input.txt and tokenize the content
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        # print(f'1 epoch = {len(self.tokens) // (self.B * self.T)} batches')
        
        # calculates the starting position in the tokenized data (self.tokens) from where this specific DDP process will begin reading data
        self.current_position = self.B * self.T * self.ddp_process_rank # Each process starts at an offset determined by its unique rank, ensuring that each process works on a different, non-overlapping portion of the dataset


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.ddp_num_processes
        # if loading the next batch would be out of bounds, rest
        if self.current_position + (B * T * self.ddp_process_rank + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.ddp_process_rank #rest back to starting 
        return x, y

#---------------- Distributed Data Parallel setup ------------#
# after executing 'torchrun' command, the env variables RANK, LOCAL_RANK, and WORLD_SIZE get set
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist # package for distributed training - that provides the low-level tools for multi-GPU and multi-node coordination
import os

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
    
#---------------- train the model ------------#
max_steps = 50
total_batch_size = 524288 # setting to ~0.5M to comply with gpt2 batchsize. Here we used a batch size that is power of two
batch_size = 16
ctx_len = 1024
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
model = torch.compile(model)
if ddp: 
  model = DDP(model, device_ids=[ddp_local_rank]) #ddp_local_rank tells DDP which specific GPU the model replica for this process should reside on
raw_model = model.module if ddp else model

# do the training
train_loader = DataLoaderLite(batch_size, ctx_len, ddp_process_rank=ddp_rank, ddp_num_processes=ddp_world_size)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time() # to capture time taken
    optimizer.zero_grad() # reset the gradients
    # --- gradient accumulation ---- #
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)

      # Enables autocasting(i.e to reducing datatype to BFLOAT16) for the forward pass (model + loss) .
      # check here what all gets converted https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
      with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
      loss = loss / grad_accum_steps # average the loss over the mini-batches - done similar to torch.nn.functional.cross_entropy that computes the mean loss over the elements in batch
      loss_accum += loss.detach()
      # we want the ddp to do the gradient sync accross gpus only at the last step of gradient accumulation 
      # This is to reduced communication overhead and improved training speed 
      if ddp:
          model.required_backward_grad_sync = (micro_step == grad_accum_steps - 1)
      loss.backward()
    # ----------------------------- #
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
    torch.cuda.synchronize() # wait until gpu is done
    t1 = time.time()
    dt = t1-t0 #calculate elasped time in sec
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size #how many tokens are we training in one step
    tokens_per_sec = tokens_processed / dt #how many token data are we training
    if master_process:
      print(f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm:{norm:.4f} | dt:{dt*1000:.2f}ms | tok/sec:{tokens_per_sec:.2f}")

if ddp: # clean up the distributed environment at the end of DDP training
  destroy_process_group()