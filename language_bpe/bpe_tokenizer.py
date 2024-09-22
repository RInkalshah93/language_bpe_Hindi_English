import sys
import regex as re
from tqdm import tqdm
from .base import Tokenizer, get_stats, merge, merge_hindi

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BPETokenizer(Tokenizer):

    def __init__(self, pattern=None, word_pattern = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.word_pattern = None
        self.compiled_pattern_word = None
        if word_pattern:
            self.word_pattern = word_pattern
            self.compiled_pattern_word = re.compile(self.word_pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def build(self, text, vocab_size, verbose=False):

        text_chunks = re.findall(self.compiled_pattern, text)

        if self.compiled_pattern_word:
            print("Spliting hindi words")
            text_chunks_words = []
            for chunk in tqdm(text_chunks):
                element_chunks = re.findall(self.compiled_pattern_word, chunk)
                if element_chunks == []:
                    text_chunks_words.append(chunk) 
                else:
                    text_chunks_words.extend(element_chunks[0]) 
            text_chunks = text_chunks_words
        
        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        vocab.update({idx: bytes(list(chr(value).encode('utf-8'))) for idx,value in zip(range(256, 384), range(2304, 2432))})

        print("Merging hindi characters in single token")
        for index in tqdm(range(256, 384)):
            pair = list(vocab[index])
            ids = [merge_hindi(chunk_ids, pair, index) for chunk_ids in ids]
        
        num_merges = vocab_size - 384

        original_length = len([x for xs in ids for x in xs])

        print("Building BPE")
        for i in tqdm(range(num_merges), file=sys.stdout):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 384 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                try:
                    tqdm.write(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx].decode('utf-8')}) had {stats[pair]} occurrences")
                except Exception as e:
                    tqdm.write(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        lenght_after_merging = len([x for xs in ids for x in xs])

        print(f'Compression ratio: {original_length/lenght_after_merging}')

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, ids):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        if self.compiled_pattern_word:
            print("Spliting hindi words")
            text_chunks_words = []
            for chunk in tqdm(text_chunks):
                element_chunks = re.findall(self.compiled_pattern_word, chunk)
                if element_chunks == []:
                    text_chunks_words.append(chunk) 
                else:
                    text_chunks_words.extend(element_chunks[0]) 
            text_chunks = text_chunks_words
        # all chunks of text are encoded separately, then results are joined
        ids_list = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            ids = list(chunk_bytes)
            vocab = {idx: bytes([idx]) for idx in range(256)}
            vocab.update({idx: bytes(list(chr(value).encode('utf-8'))) for idx,value in zip(range(256, 384), range(2304, 2432))})
            for index in tqdm(range(256, 384)):
                pair = list(vocab[index])
                ids = merge_hindi(ids, pair, index)
            chunk_ids = self._encode_chunk(ids)
            ids_list.extend(chunk_ids)
        return ids_list

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids