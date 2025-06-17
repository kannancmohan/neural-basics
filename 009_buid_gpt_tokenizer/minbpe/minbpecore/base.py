def get_consecutive_pairs(ids, consec_pairs=None):
    """ 
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    consec_pairs = {} if consec_pairs is None else consec_pairs
    for pair in zip(ids, ids[1:]):
        consec_pairs[pair] = consec_pairs.get(pair, 0) + 1
    return consec_pairs

def replace_pair(ids, replaceable_pair, replace_with_id):
    """ 
    In the list of integers (ids), replace all consecutive occurrences
    of replaceable_pair with the new integer token replace_with_id
    Example: ids=[1, 2, 3, 1, 2], replaceable_pair=(1, 2), replace_with_id=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == replaceable_pair[0] and ids[i+1] == replaceable_pair[1]:
            new_ids.append(replace_with_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# # -----------------------------------------------------------------------------
# # the base Tokenizer class

# class Tokenizer:
#     """Base class for Tokenizers"""

#     def __init__(self):
#         # default: vocab size of 256 (all bytes), no merges, no patterns
#         self.merges = {} # (int, int) -> int
#         self.pattern = "" # str
#         # self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
#         self.vocab = self._build_vocab() # int -> bytes

#     def train(self, text, vocab_size, verbose=False):
#         # Tokenizer can train a vocabulary of size vocab_size from text
#         raise NotImplementedError

#     def encode(self, text):
#         # Tokenizer can encode a string into a list of integers
#         raise NotImplementedError

#     def decode(self, ids):
#         # Tokenizer can decode a list of integers into a string
#         raise NotImplementedError

#     def _build_vocab(self):
#         # vocab is simply and deterministically derived from merges
#         vocab = {idx: bytes([idx]) for idx in range(256)}
#         for (p0, p1), idx in self.merges.items():
#             vocab[idx] = vocab[p0] + vocab[p1]
#         # for special, idx in self.special_tokens.items():
#         #     vocab[idx] = special.encode("utf-8")
#         return vocab