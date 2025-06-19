import math
from dataclasses import dataclass
import torch
import torch.nn as nn
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
    