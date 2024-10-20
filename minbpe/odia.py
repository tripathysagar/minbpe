import regex as re
from typing import Dict, List, Set, Union

from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Odia:
    # List of valid Unicode ranges for Odia script
    valid_ranges = {
        'anusvara': range(0x0B01, 0x0B03 + 1),
        'matras': set(range(0x0B3C, 0x0B4D + 1)).union({0x0B55, 0x0B56, 0x0B57}),
        'digits': range(0x0B66, 0x0B6F + 1),
        'sign': set(range(0x0B72, 0x0B77 + 1)).union(set([0x2018, 0x2019, 0x201C, 0x201D])),
        'aux_sign': {0x0B70, 0x0964, 0x0965},
        'vowels': range(0x0B05, 0x0B14 + 1),
        'consonants': set(range(0x0B15, 0x0B39 + 1)).union({0x0B5F, 0x0B71}),
    }

    # List of Unicode code points to ignore
    ignore_case = {0x0B0D, 0x0B0E, 0x0B11, 0x0B12, 0x0B29, 0x0B31, 0x0B34, 0x0B45, 0x0B46, 0x0B5E, 0x0B49, 0x0B4A}

    def __init__(self):
        self.odia_chars = {
            key: [chr(i) for i in (val if isinstance(val, range) else val) if i not in self.ignore_case]
            for key, val in self.valid_ranges.items()
        }

        self.odia_chars['complex_char'] = []
        for i in self.odia_chars['matras'] + self.odia_chars['anusvara']:
            for j in self.odia_chars['vowels'] + self.odia_chars['consonants']:
                self.odia_chars['complex_char'].append(''.join([j, i]))


    def generate_odia_pattern(self):
        """
        Generate a regex pattern that matches any valid Odia character.
        """
        pattern_parts = []
        for key, chars in list(self.odia_chars.items())[::-1]: 
            # we are traversing in the revese order for finding the complex char first, consonants, vowels etc 
            # Each character set will be part of the pattern, using `|` for alternation
            if key == 'complex_char':
                # Complex chars are added as whole sequences, so no need for []
                pattern_parts.append('|'.join(re.escape(char) for char in chars))
            else:
                # Individual chars are added inside []
                pattern_parts.append('[' + ''.join(re.escape(char) for char in chars) + ']')


        # Join all the parts with alternation and return the pattern
        return '|'.join(pattern_parts)

    def get_chars(self, category: str) -> List[str]:
        if category not in self.odia_chars:
            raise ValueError(f"Invalid category: {category}")
        return self.odia_chars[category]

    def __getattr__(self, name: str) -> List[str]:
        if name in self.odia_chars:
            return self.get_chars(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class OdiaRegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.odia = Odia()  # Initialize the Odia character support
        odia_pattern = self.odia.generate_odia_pattern()  # Generate Odia pattern
        base_pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # Append the Odia-specific pattern to the base pattern
        self.pattern = f"{odia_pattern}|{base_pattern}"
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        print(text_chunks)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
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

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
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
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
