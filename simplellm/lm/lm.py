# Various implementation of a simple character level language model

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

#############################################################
## Naive Bi-gram Model (Counting)
#############################################################

class BiGramModel:
    def __init__(self, sample_sep:str = ".", dummy_count:int = 1) -> None:
        # KEY IDEA: mark start and ending of a sample by "."
        self.sample_sep: str = sample_sep
        # KEY IDEA: add dummy count to avoid zero probability (smoothing)
        self.dummy_count: int = dummy_count
        self.characters: list[str] = [sample_sep]
        
        # N*N matrices where N is the number of characters
        # bigram_prob[i, j] = P(next character is char_j | current character is char_i)
        self.bigram_count: torch.Tensor = torch.zeros((1,1))
        self.c2i: dict[str, int] = {sample_sep: 0}

    def train(self, samples: list[str]) -> None:
        # build the character set
        all_text = "".join(samples)
        self.characters = [self.sample_sep] + list(sorted(set(all_text)))
        self.c2i = {c: i for i, c in enumerate(self.characters)}
        # print(f"c2i: {self.c2i}")

        # build the bigram count
        char_set_size = len(self.characters)
        self.bigram_count = torch.zeros((char_set_size, char_set_size)) + self.dummy_count
        for sample in samples:
            full_sample = "." + sample + "."
            for i in range(len(full_sample) - 1):
                c = full_sample[i]
                c_next = full_sample[i + 1]
                self.bigram_count[self.c2i[c], self.c2i[c_next]] += 1
        
    def generate(self, n_samples: int = 10, max_length: int = 100) -> list[str]:
        samples: list[str] = []
        
        # normalize the count to get probability
        bigram_prob = self.bigram_count / self.bigram_count.sum(1, keepdim=True)

        for _ in range(n_samples):
            current_char = self.sample_sep
            generated_text = ""
            for _ in range(max_length):
                next_char = self._sample_next_char(bigram_prob, current_char)
                if next_char == ".":
                    break
                generated_text += next_char
                current_char = next_char
            samples.append(generated_text)
        return samples

    def _sample_next_char(self, bigram_prob: torch.Tensor, current_char: str) -> str:
        idx = self.c2i[current_char]
        next_char_idx = int(torch.multinomial(bigram_prob[idx], 1).item())
        return self.characters[next_char_idx]

