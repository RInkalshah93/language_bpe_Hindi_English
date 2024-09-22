"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
import argparse
from language_bpe import BPETokenizer

parser = argparse.ArgumentParser(prog="BPE Tokenizer")
parser.add_argument("--input_file", default="data/hindi.txt", type=str)
parser.add_argument("--output_file", default="tokenizer", type=str)
parser.add_argument("--vocab_size", default=500, type=int)
parser.add_argument("--is_english", action=argparse.BooleanOptionalAction)
args = parser.parse_args()


# open some text and train a vocab of 512 tokens
text = open(args.input_file, "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

HINDI_SPLIT_PATTERN = r'[^\r\n\p{L}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+'
HINDI_WORD_SPLIT_PATTERN = r'([\s\p{L}\p{M}]{2,})([कगतन](?:\p{M}))$'

t0 = time.time()
# construct the Tokenizer object and kick off verbose training
if args.is_english:
    tokenizer = BPETokenizer()
else:
    tokenizer = BPETokenizer(pattern=HINDI_SPLIT_PATTERN, word_pattern=HINDI_WORD_SPLIT_PATTERN)

tokenizer.build(text, args.vocab_size, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", args.output_file)
tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")