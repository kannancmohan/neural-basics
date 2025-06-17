import regex as re
from .base import get_consecutive_pairs, replace_pair

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:
    
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # split the text using the regex
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # convert chunks to byte stream
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        merged_pairs = {} # dict of (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # dict of int -> bytes
        for i in range(num_merges):
            
            consec_pairs = {}
            for chunk_ids in ids:
                get_consecutive_pairs(chunk_ids, consec_pairs)
                
            top_pair = max(consec_pairs, key=consec_pairs.get) # find the pair with the highest count
            
            new_idx = 256 + i
            ids = [replace_pair(chunk_ids, top_pair, new_idx) for chunk_ids in ids]
            
            # add the merged pair and its new id to merged_pairs
            merged_pairs[top_pair] = new_idx
            
            # add the merged pairs id and its concatenated byte sequences to vocab dict
            # here we concat two byte sequences (vocab[p0] + vocab[p1]), which is useful when decoding 
            # for eg, consider we merged a pair (101, 32) with id 257, The bytes for pairs are b'e' and b' ' respectively, then concatenating these two bytes b'e' + b' ' = b'e '
            vocab[new_idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {new_idx} ({vocab[new_idx]}) had {consec_pairs[top_pair]} occurrences")

        
        self.merges = merged_pairs # used in encode()
        self.vocab = vocab   # used in decode()
        
    def encode(self, text):
         # given a string text, return the byte ids
        text_bytes = text.encode("utf-8") # # encode text to raw byte-stream
        ids = list(text_bytes)# convert byte stream to its numerical equivalent
        
        while len(ids) >= 2:
            # print(ids)
            consec_pairs = get_consecutive_pairs(ids)
            # Find the first pair that exists in our merges
            first_common_pair = next((k for k in consec_pairs if k in self.merges), None)
            if first_common_pair is None:
                break
            ids = replace_pair(ids, first_common_pair, self.merges[first_common_pair])
        return ids
    
    def decode(self, ids):
         # given ids (list of integers), return string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text