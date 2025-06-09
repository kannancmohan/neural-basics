import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000 # training iteration
eval_interval = 300 # interval at which we evaluate the loss on train and val sets during training
learning_rate = 1e-2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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
  ix = torch.randint(len(data) - block_size, (batch_size,)) # generates batch_size (4) random integers.
  # generate input and output sequence based on above 4 random integers
  x = torch.stack([data[i:i+block_size] for i in ix]) # rows of input sequence
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # rows of output sequence
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

# simple Pytorch Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # creates a lookup table that maps each input character to a vector of scores/weights (logits) representing how likely each character is to follow it. 
        # The scores are randomly initialized at first and are learned during training.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # shape of input idx=[4,8] and target=[4,8]
        # lookup against the input to get the corresponding logits 
        logits = self.token_embedding_table(idx) # (B-Batch,T-Times,C-Channel)

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
            logits, loss = self(idx) # calls 'forward' method to get the predictions. 'loss' can be ignored
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
model = BigramLanguageModel(vocab_size)
m = model.to(device)

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

#lets see what model can generate after training 
context = torch.zeros((1, 1), dtype=torch.long, device=device) # basically, 0 is char '\n'. we start with newline to predict the next char and so on
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))