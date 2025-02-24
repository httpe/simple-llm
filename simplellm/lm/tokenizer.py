#############################################################
## Tokenization
#############################################################

class CharacterTokenizer:
    def __init__(self) -> None:
        self.start_sym = "^"
        self.end_sym = "$"
        self.pad_sym = "."
        self.start_token = 1
        self.end_token = 2
        self.pad_token = 0
        self.c2i = {self.start_sym: self.start_token, self.end_sym: self.end_token, self.pad_sym: self.pad_token}
        self.i2c = {self.start_token: self.start_sym, self.end_token: self.end_sym, self.pad_token: self.pad_sym}

    def train(self, samples: list[str]):
        # build the character set
        # KEY IDEA: naive tokenization, we do per character tokenization
        vocabulary = sorted(set("".join(samples)))
        assert self.start_sym not in vocabulary
        assert self.end_sym not in vocabulary
        assert self.pad_sym not in vocabulary
        next_token_idx = max(self.i2c.keys()) + 1
        for c in vocabulary:
            if c in self.c2i:
                continue
            self.c2i[c] = next_token_idx
            next_token_idx += 1
        
        self.i2c = {i: c for c, i in self.c2i.items()}

    def encode(self, sample: str) -> list[int]:
        tokens = [self.c2i[c] for c in sample]
        return tokens
    def decode(self, tokens: list[int]) -> str:
        chars = [self.i2c[i] for i in tokens]
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.c2i)