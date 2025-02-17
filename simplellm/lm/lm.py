# Various implementation of a simple character level language model

import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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

#############################################################
## Naive Tri-gram Model (Counting)
#############################################################

class TriGramModel:
    def __init__(self, sample_sep:str = ".", dummy_count:int = 1) -> None:
        # KEY IDEA: mark start and ending of a sample by "."
        self.sample_sep: str = sample_sep
        # KEY IDEA: add dummy count to avoid zero probability (smoothing)
        self.dummy_count: int = dummy_count
        self.characters: list[str] = [sample_sep]
        
        # N*N matrices where N is the number of characters
        # bigram_prob[i, j] = P(next character is char_j | current character is char_i)
        self.trigram_count: torch.Tensor = torch.zeros((1,1,1))
        self.c2i: dict[str, int] = {sample_sep: 0}

    def train(self, samples: list[str]) -> None:
        # build the character set
        all_text = "".join(samples)
        self.characters = [self.sample_sep] + list(sorted(set(all_text)))
        self.c2i = {c: i for i, c in enumerate(self.characters)}
        # print(f"c2i: {self.c2i}")

        # build the tri-gram count
        char_set_size = len(self.characters)
        self.trigram_count = torch.zeros((char_set_size, char_set_size, char_set_size)) + self.dummy_count
        for sample in samples:
            full_sample = ".." + sample + "."
            for i in range(len(full_sample) - 2):
                c = full_sample[i]
                c_next = full_sample[i + 1]
                c_next_next = full_sample[i + 2]
                self.trigram_count[self.c2i[c], self.c2i[c_next], self.c2i[c_next_next]] += 1
        
    def generate(self, n_samples: int = 10, max_length: int = 100) -> list[str]:
        samples: list[str] = []
        
        # normalize the count to get probability
        trigram_prob = self.trigram_count / self.trigram_count.sum(2, keepdim=True)

        for _ in range(n_samples):
            context = (self.sample_sep, self.sample_sep)
            generated_text = ""
            for _ in range(max_length):
                next_char = self._sample_next_char(trigram_prob, context)
                if next_char == ".":
                    break
                generated_text += next_char
                context = (context[1], next_char)
            samples.append(generated_text)
        return samples

    def _sample_next_char(self, trigram_prob: torch.Tensor, context: tuple[str, str]) -> str:
        idx = (self.c2i[context[0]], self.c2i[context[1]])
        prob = trigram_prob[idx]
        next_char_idx = int(torch.multinomial(prob, 1).item())
        return self.characters[next_char_idx]
    


#############################################################
## Neural N-gram Model
#############################################################

class TensorBiGramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(TensorBiGramModel, self).__init__()

        self.vocab_size = vocab_size

        # this will be trained into the bi-gram counting matrix
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, x):
        # x: (batch_size, 1), int tensor between 0 and vocab_size - 1
        embed = self.embedding(x) # (batch_size, 1, vocab_size)
        
        bi_gram_logits = embed.view(embed.size(0), -1) # (batch_size, vocab_size)

        # returning the logits, it is the log(count_of_next_char_given_current_char) in the counting model
        return bi_gram_logits


class NeuralNGramModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, look_back: int, hidden_sizes: list[int], output_bias=True):
        super(NeuralNGramModel, self).__init__()

        # for N-gram model, we need to look back n_gram - 1 characters
        self.look_back = look_back

        self.vocab_size = vocab_size

        # KEY IDEA: convert input characters to embedding vectors
        self.embedding = nn.Embedding(self.vocab_size, embed_size)

        layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layer_input_size = self.look_back * embed_size
            else:
                layer_input_size = hidden_sizes[i-1]
            layers.extend([nn.Linear(layer_input_size, size), nn.ReLU()])

        if len(hidden_sizes) == 0:
            output_layer_input_size = self.look_back * embed_size
        else:
            output_layer_input_size = hidden_sizes[-1]
        layers.append(nn.Linear(output_layer_input_size, self.vocab_size, bias=output_bias))

        self.layers = layers
        self.layers_stack = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, look_back), int tensor between 0 and vocab_size - 1
        embed = self.embedding(x) # (batch_size, look_back, embed_size)

        # flatten
        nn_input = embed.view(embed.size(0), -1) # (batch_size, look_back * embed_size)
        nn_output_logits = self.layers_stack(nn_input) # (batch_size, vocab_size)

        return nn_output_logits

#############################################################
## Tokenization
#############################################################

class CharacterTokenizer:
    def __init__(self, sample_sep: str = ".") -> None:
        self.sample_sep = sample_sep
        self.c2i = {self.sample_sep: 0}
        self.i2c = {0: self.sample_sep}

    def train(self, samples: list[str]):
        # build the character set
        # KEY IDEA: naive tokenization, we do per character tokenization
        vocabulary = sorted(set("".join(samples)))
        assert self.sample_sep not in vocabulary
        for c in vocabulary:
            if c in self.c2i:
                continue
            self.c2i[c] = len(self.c2i)
        
        self.i2c = {i: c for c, i in self.c2i.items()}

    def tokenize(self, sample: str) -> list[int]:
        return [self.c2i[c] for c in sample]

    @property
    def vocab_size(self) -> int:
        return len(self.c2i)
    
    def char_to_token(self, char: str) -> int:
        return self.c2i[char]

    def token_to_char(self, token: int) -> str:
        return self.i2c[token]

#############################################################
## Neural Model Training
#############################################################

def prepare_torch_dataset(samples: list[str], look_back: int, sample_sep: str = ".") -> tuple[CharacterTokenizer, TensorDataset]:
    # build the vocabulary
    tokenizer = CharacterTokenizer(sample_sep)
    tokenizer.train(samples)

    # convert samples to torch tensors
    X = []
    Y = []
    for sample in samples:
        full_sample = sample_sep * look_back + sample + sample_sep
        tokens = tokenizer.tokenize(full_sample)
        for i in range(look_back, len(tokens)):
            X.append(tokens[i-look_back:i])
            Y.append(tokens[i])

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, Y)

    return tokenizer, dataset


def train_torch_lm_model(model: nn.Module, dataset: TensorDataset, batch_size=32, epochs=1000, learning_rate=0.01):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_samples = len(dataloader.dataset) # type: ignore
    
    # Initialize model, loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in dataloader:
            batch_size = X_batch.shape[0]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size

            # print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Batch Loss: {loss.item():.4f} X[0] = {X_batch[0]}, Y[0] = {y_batch[0]}')

        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
            
    return model


def sample_torch_lm_model(model: nn.Module, tokenizer: CharacterTokenizer, look_back: int, n_samples: int = 10, max_length: int = 100, sample_sep: str = ".") -> list[str]:
    samples: list[str] = []

    for _ in range(n_samples):
        current_tokens = [tokenizer.char_to_token(sample_sep)] * look_back
        generated_text = ""

        for _ in range(max_length):
            current_tokens_tensor = torch.tensor(current_tokens, dtype=torch.long).unsqueeze(0) # (1, look_back)

            with torch.no_grad():
                logits = model(current_tokens_tensor) # (1, vocab_size)

            # randomly pick one token out based on the logits distribution
            next_token = int(torch.multinomial(torch.softmax(logits[0], dim=0), 1).item())
            next_char = tokenizer.token_to_char(next_token)

            if next_char == sample_sep:
                break

            generated_text += next_char
            current_tokens = current_tokens[1:] + [next_token]
        
        samples.append(generated_text)

    return samples

