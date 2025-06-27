"""
HellaSwag: Can a Machine Really Finish YourSentence?

This dataset is a benchmark for evaluating a model's ability to perform commonsense natural language inference.
The task is to predict the most plausible ending to a given context from four possible choices.
While easy for humans (around 95% accuracy), it's challenging for state-of-the-art models.

A single example from the dataset looks like this:
{
  "ctx": "A woman is outside with a bucket and a dog. The dog is running around the bucket. The woman then",
  "endings": [
    "puts the bucket on her head.",
    "pours the bucket of water on the dog.",
    "gets inside the bucket.",
    "jumps on the dog."
  ],
  "label": 1
}
The 'label' indicates the correct ending, which is "pours the bucket of water on the dog."in this case.

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)
"""

import os
import requests
import json
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), "hellaswag")
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# download and get examples from Hellaswag 
def iterate_hellaswag_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example
def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# render/transform example from Hellaswag
enc = tiktoken.get_encoding("gpt2")
def render_example(example):
    """
    Prepares a single HellaSwag example for a language model.
    Args:
        example (dict): A dictionary containing a single HellaSwag instance,
                        with keys "ctx" (context string), "endings" (list of
                        possible completion strings), and "label" (index of
                        the correct ending).
    Returns:
        tuple: A tuple containing:
            - data (dict): Original example data with tokenized context and endings.
            - tokens (torch.Tensor): A tensor of shape (4, max_len) containing
                                     the tokenized context and endings for all 4 choices,
                                     padded to the maximum length.
            - mask (torch.Tensor): A tensor of shape (4, max_len) where 1s indicate
                                   the tokens corresponding to the endings (for evaluation)
                                   and 0s indicate the context tokens and padding.
            - label (int): The index of the correct ending.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    tok_rows = [] # to capture encoded ctx + encoded ending 
    mask_rows = [] # to capture masking of ctx(masked as 0) and ending(masked as 1)
    ctx_tokens = enc.encode(ctx) # encode ctx token
    data["ctx_tokens"] = ctx_tokens # set the encoded to data dict
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)
    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    return data, tokens, mask, label


# This function is a core part of evaluating the language model's performance
# on the HellaSwag dataset. HellaSwag is a multiple-choice benchmark where the
# task is to choose the most plausible ending for a given context from four options.
#
# The `get_most_likely_row` function takes the model's output (logits) for the
# four possible sequences (context + each ending) and determines which ending
# the model considers most likely. It does this by calculating the autoregressive
# loss *specifically* over the tokens that constitute the "ending" part of each
# sequence.
#
# Here's a breakdown of the process within this function:
# 1.  **Aligning Logits and Tokens:** The model's logits at position `i` predict
#     the token at position `i+1`. To calculate the loss correctly, we shift the
#     logits one position back and the target tokens (from the input sequences)
#     one position forward. This ensures that each logit is compared against the
#     actual next token it was meant to predict.
# 2.  **Calculating Per-Token Loss:** The cross-entropy loss is computed between
#     the shifted logits and the shifted tokens for every single token position
#     in each of the four sequences.
# 3.  **Masking Context Loss:** A `mask` is used to distinguish between the context
#     tokens and the ending tokens. This mask is applied to the per-token losses,
#     effectively setting the loss for all context tokens to zero. This means we
#     only consider the loss incurred by the model when predicting the tokens
#     that make up the four different endings.
# 4.  **Averaging Ending Loss:** The losses for the ending tokens are summed up
#     for each of the four sequences. This sum is then divided by the total number
#     of ending tokens in that specific sequence to get an *average* loss for the
#     ending. This provides a single score representing how "surprised" the model
#     was by the actual tokens in that ending, given the preceding context.
# 5.  **Selecting the Most Likely Ending:** The ending with the lowest average loss
#     is considered the most likely completion according to the model. A lower loss
#     corresponds to a higher probability assigned by the model to that sequence
#     of tokens. The function returns the index (0, 1, 2, or 3) of this most likely ending.
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    
    ## Calculate Shifted Logits and Tokens - done to align the model's prediction for a token with the actual next token. 
    # shift_logits will contain the logits predicting the next token.
    # selects all the logits for all batches, for all tokens in the sequence except the very last one and for all vocabulary entries
    shift_logits = (logits[..., :-1, :]).contiguous() # here logits[..., :-1, :] equivalent to logits[:, :-1, :]. shape of shift_logits (B, T-1, vocab_size)
    # shift_tokens will contain the actual next tokens(this can be considered as the labels(y) we normally use for calculating the cross_entropy loss)
    # select all tokens for all batches and for all tokens elements from index 1(i.e starting from second token) up to the end 
    shift_tokens = (tokens[..., 1:]).contiguous() #shape of shift_tokens (B, T-1)
    
    ## Reshape Logits and Tokens tensors for cross_entropy calculation ##
    # reshape (B, T-1, vocab_size) to (B*T-1, vocab_size)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    # reshape (B, T-1) to (B*T-1). reshaping so as cross_entropy expects 1D teensor for target labels
    flat_shift_tokens = shift_tokens.view(-1)

    ## Calculating Loss per Token  ##
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1) # reshape 1D-(B * T-1) to 2D-(B ,T-1)

    ## Masking the Loss ##
    # Since the mask is 0 for context tokens and 1 for ending tokens, this operation effectively zeros out the loss for the context tokens, leaving only the loss for the ending tokens
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask

    ## Averaging Loss over the Ending tokens##
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    
    ## Selecting the Best Ending ## 
    # now we have a loss for each of the 4 completions. the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm