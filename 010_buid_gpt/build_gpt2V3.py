# Improvements
# Add loss calculation to models forward method
# pass in Labels(y) also as an input to the model
# train the model with a small dataset 

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import tiktoken
from torch.nn import functional as F

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

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
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
        # self.c_proj.NANOGPT_SCALE_INIT = 1

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
        # self.c_proj.NANOGPT_SCALE_INIT = 1
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
        # compute attention scores ("affinities")
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # auto aggressive mask - which make suer that the tokens attends only to token before them
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # causal mask
        att = F.softmax(att, dim=-1) # normalizes so that attention sums to 1
        y = att @ v # perform the weighted aggregation of the values
        # concat the results of all head 
        # here transpose(1, 2) swaps the dimensions . ie from (B, nh, T, hs) to (B, T, nh, hs)
        # contiguous - This method ensures that the tensor is stored contiguously in memory. Transposing a tensor can sometimes result in a non-contiguous memory layout, which can cause issues or performance penalties with subsequent operations like view. Calling contiguous() creates a new tensor with the same data but a contiguous memory arrangement if it wasn't already contiguous.
        # .view(B, T, C) - This reshapes the tensor back into the original embedding dimension shape. The values from the nh heads, each with hs dimensions, are laid out side-by-side to form a single vector of size C
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y      
    
    
#---------------- train the model with a small dataset ------------#
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
print(f'{device=}')
batch_size = 4
ctx_len = 32
learning_rate = 3e-4

# get the contents of input.txt
with open('input.txt', 'r') as f:
    text = f.read()

# create the inputs(x) and labels(y)
text = text[:10000] # taking only first 10000
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)
print(f'{len(tokens)=}')
buf = torch.tensor(tokens[:batch_size*ctx_len + 1])
buf = buf.to(device) # here we need to reassign back to buf, so that it refers to the new tensor object created in GPU

x = buf[:-1].view(batch_size, ctx_len)
y = buf[1:].view(batch_size, ctx_len)

# init custom model
model = GPT(GPTConfig())
model.to(device)

# do the training 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # create a PyTorch optimizer - here we using AdamW instead of SGD optimizer
for steps in range(50):
    optimizer.zero_grad() # reset the gradients
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'{steps=}, {loss.item()}')