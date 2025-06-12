import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
ctx_len = 256 # what is the maximum context length for predictions?
max_iters = 5000 # training iteration
eval_interval = 500 # interval at which we evaluate the loss on train and val sets during training
learning_rate = 3e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # dim of input embedding

n_head = 6 # no of attention heads in multi-head attention
n_layer = 6 # no of transformer blocks
dropout = 0.2 # 

torch.manual_seed(1337) # ensures random number generation is reproducible

# read the input file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# construct vocab 
chars = sorted (list(set(text))) # extract all unique chars from entire text
vocab_size = len(chars)

# construct encoder and decoder
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] #encoder: take a string, output a list of integers
decode = lambda ix: ''.join([itos[i] for i in ix]) # decoder: take a list of integers, output a string

# tokenization/encode the entire text to integer ids
data = torch.tensor(encode(text), dtype=torch.long)

## Build training & validation split from encoded data ##
n = int(0.9 * len(data)) # first 90% will be used for train and remaining 10% for validation
train_data = data[:n]
val_data = data[n:]

# method to generate random batch of inputs and target data.
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - ctx_len, (batch_size,)) # generates batch_size (4) random integers.
  # generate input and output sequence based on above 4 random integers
  x = torch.stack([data[i:i+ctx_len] for i in ix]) # rows of input sequence
  y = torch.stack([data[i+1:i+ctx_len+1] for i in ix]) # rows of output sequence
  x, y = x.to(device), y.to(device) # use cuba 
  return x, y

# function helps you periodically assess how well your model is performing on training & validation data
# This is crucial for monitoring training progress and detecting overfitting
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # Calculates the average loss over all iterations for the current split ('train' or 'val')
    model.train()
    return out

class Head(nn.Module):
      """ one head of self-attention """

      def __init__(self, head_size):
          super().__init__()
          #init query,key and value
          self.key = nn.Linear(n_embd, head_size, bias=False)
          self.query = nn.Linear(n_embd, head_size, bias=False)
          self.value = nn.Linear(n_embd, head_size, bias=False)
          # creates a lower triangular matrix of ones - register it as buffer as they are not trainable parameters
          self.register_buffer('tril', torch.tril(torch.ones(ctx_len, ctx_len)))
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          B,T,C = x.shape
          # forward through query,key and value
          k = self.key(x)   # (B,T,head_size)
          q = self.query(x) # (B,T,head_size)
          v = self.value(x) # (B,T,head_size)
          # compute attention scores ("affinities")
          wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
          wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
          wei = F.softmax(wei, dim=-1) # (B, T, T)
          wei = self.dropout(wei)
          # perform the weighted aggregation of the values
          out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
          return out

class MultiHeadAttention(nn.Module):
      """ multiple heads of self-attention in parallel """

      def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # added for skip connection - this layer is just to transform  the dimension of the concatenated outputs from all the attention heads back to the original input's embedding dimension 
        self.proj = nn.Linear(num_heads * head_size, n_embd) # a linear layer
        self.dropout = nn.Dropout(dropout)
      
      def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concat the results of all head by its last dim
        out = self.proj(out) # linear layer - transforms dim (B, T, num_heads * head_size) to (B,T, n_embd)
        return out

class FeedForward(nn.Module):
      """ a simple linear layer followed by a non-linearity """

      def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
          # Standard practice of multiplying by 4 -  a common technique to give the FeedForward network more capacity to process the contextually enriched features
          nn.Linear(n_embd, 4 * n_embd),
          nn.ReLU(),
          nn.Linear(4 * n_embd, n_embd),# added for skip connection - allows the model to learn the best way to linearly combine these non-linear features
          nn.Dropout(dropout),
        )

      def forward(self, x):
        return self.net(x)

class Block(nn.Module):
      """ Transformer block: communication followed by computation """

      def __init__(self, n_embd, n_head):
          # n_embd: embedding dimension, n_head: the number of heads we'd like
          super().__init__()
          head_size = n_embd // n_head # 32/4 --> 8
          # the communcation
          self.sa = MultiHeadAttention(n_head, head_size)
          # the computation
          self.ffwd = FeedForward(n_embd)
          # the LayerNorm layers
          self.ln1 = nn.LayerNorm(n_embd)
          self.ln2 = nn.LayerNorm(n_embd)

      def forward(self, x):
          # here is the skip/residual connection - where we add the input to the output of the layer
          # also, we are invoking the LayerNorm layers(self.ln1(x) and self.ln2(x) ) before the skip/residual calculation
          x = x + self.sa(self.ln1(x))
          x = x + self.ffwd(self.ln2(x))
          return x

# simple Pytorch Bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # input embedding layer
        self.position_embedding_table = nn.Embedding(ctx_len, n_embd) # position embedding layer
        # self.blocks = nn.Sequential(
        #   Block(n_embd, n_head=4),
        #   Block(n_embd, n_head=4),
        #   Block(n_embd, n_head=4),
        #   nn.LayerNorm(n_embd), # final LayerNorm - add it after transformer and before the final layer
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm - add it after the transformer and before the final layer
        self.lm_head = nn.Linear(n_embd, vocab_size) # hidden layer. lm_head means language modeling head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B-Batch,T-Times,C-Channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        ## feeding to blocks of multi-head attention and its compuation
        x = self.blocks(x)
        logits = self.lm_head(x)  

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshaping to conform with pytorch cross_entropy. i.e [4,8,65] to [32,65]
            targets = targets.view(B*T) # reshaping to conform with pytorch cross_entropy. i.e [4,8] to [32]
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size token bcoz the model model predicts one new token at a time and appends it to the idx sequence. As new tokens are added, the idx sequence grows longer than ctx_len
            idx_cond = idx[:, -ctx_len:] 

            logits, loss = self(idx_cond) # calls 'forward' method to get the predictions. 'loss' can be ignored
            # focus only on the last time(T) step. -1 extracts the last element along the specified dimension(here its time dim 'T')
            logits = logits[:, -1, :] #  becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # concatenates the newly sampled token IDs (idx_next) to the end of the current sequence (idx) along the time dimension (dim=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) . T dimension size increase over the iteration.
        return idx

# init the model
model = BigramLanguageModel()
m = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params} or {total_params/1e6}Mil")

# create a PyTorch optimizer - here we using AdamW instead of SGD optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# train the model
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train') # sample a batch of data

    logits, loss = model(xb, yb) # evaluate the loss
    optimizer.zero_grad(set_to_none=True) # zero grad
    loss.backward() # backpass
    optimizer.step() # update parameters
print(loss.item())

# Save the entire model (architecture + weights)
torch.save(model.state_dict(), "transformer_weights.pt")

#lets see what model can generate after training 
context = torch.zeros((1, 1), dtype=torch.long, device=device) # basically, 0 is char '\n'. we start with newline to predict the next char and so on
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))