"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

# pip install -q datasets # huggingface library for accessing, processing, and sharing datasets
# pip install -q tqdm # for showing progress bar
# pip install -q --upgrade datasets

import os
from datasets import load_dataset
import datasets
import tiktoken
import multiprocessing as mp
import numpy as np
from tqdm import tqdm


## create the local cache directory if it doesn't exist yet ##
local_dir = "edu_fineweb10B"
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True) # DATA_CACHE_DIR specify the directory where datasets will be cached when using datasets library
print(DATA_CACHE_DIR)

## download the dataset from huggingface ##
print("huggingface cache directory", datasets.config.HF_DATASETS_CACHE)
remote_name = "sample-10BT"
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")# downloading only the training portion of the dataset, instead of all available splits.


## init the tokenizer ##
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """ Saves the numpy array of tokens to a file"""
    np.save(filename, tokens_np)
## process the downloaded dataset by tokenizing its documents and saving them into multiple binary files (shards) ##
# The dataset is split into shards of approximately 100M tokens each.
# The first shard (shard_index 0) is designated as the validation set, and all subsequent shards are for training.
def main():
    shard_size = int(1e8) # 100M tokens per shard,so total of 100 shards
    nprocs = max(1, os.cpu_count()//2) # Use all available cores, but leave one free for the OS and main process
    print(f"Using {nprocs} processes for tokenization.")
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
        if progress_bar is not None:
            progress_bar.close()
            
if __name__ == "__main__":
    # It's good practice to explicitly set the start method if you encounter issues,
    # especially on macOS or Windows, or in environments like Nix where 'fork' might be restricted.
    # 'spawn' is generally safer and more portable.
    # mp.set_start_method("spawn", force=True) # Uncomment if you face persistent issues
    main()