from .base import get_consecutive_pairs, replace_pair

class BasicTokenizer:
    
    # def __init__(self):
    #     pass
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # input text preprocessing
        text_bytes = text.encode("utf-8") # # encode text to raw byte-stream
        ids = list(text_bytes)# convert byte stream to its numerical equivalent
        
        
        merged_pairs = {} # dict of (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # dict of int -> bytes
        for i in range(num_merges):
            consec_pairs = get_consecutive_pairs(ids)
            top_pair = max(consec_pairs, key=consec_pairs.get)# find the pair with the highest count
            
            new_idx = 256 + i
            ids = replace_pair(ids, top_pair, new_idx)
            
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