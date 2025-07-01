"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Downloads and tokenizes the data and saves data shards to disk.

Usage:
    python fineweb.py --output_dir edu_fineweb10B --shard_size 1e8 --num_procs 4
"""

import os
import argparse
import logging
from typing import Any
from datasets import load_dataset
import tiktoken
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# ------------------- Configuration -------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Download and shard FineWeb-Edu dataset.")
    parser.add_argument('--output_dir', type=str, default="edu_fineweb10B", help="Directory to save shards")
    parser.add_argument('--shard_size', type=float, default=1e8, help="Tokens per shard")
    parser.add_argument('--num_procs', type=int, default=max(1, os.cpu_count() // 2), help="Number of processes")
    parser.add_argument('--dataset_name', type=str, default="sample-10BT", help="FineWeb-Edu dataset version")
    return parser.parse_args()

# ------------------- Logging Setup -------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------- Tokenization -------------------

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc: Any) -> np.ndarray:
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    return tokens_np

def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    np.save(filename, tokens_np)

# ------------------- Sharding Logic -------------------

def process_dataset(fw, output_dir: str, shard_size: int, num_procs: int, tokenize_fn):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving shards to: {output_dir}")
    logging.info(f"Using {num_procs} processes for tokenization.")

    with mp.Pool(num_procs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize_fn, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
                remainder_len = shard_size - token_count
                all_tokens_np[token_count:token_count+remainder_len] = tokens[:remainder_len]
                write_datafile(filename, all_tokens_np)

                if progress_bar is not None:
                    progress_bar.update(remainder_len)
                    progress_bar.close()

                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder_len] = tokens[remainder_len:]
                token_count = len(tokens)-remainder_len

        # Write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
        if progress_bar is not None:
            progress_bar.close()

# ------------------- Main -------------------

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    logging.info("Loading dataset from HuggingFace...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=args.dataset_name, split="train")
    logging.info("Dataset loaded.")

    process_dataset(
        fw=fw,
        output_dir=args.output_dir,
        shard_size=int(args.shard_size),
        num_procs=args.num_procs,
        tokenize_fn=tokenize
    )
    logging.info("Sharding complete.")

if __name__ == "__main__":
    main()