# Various implementation of a simple character level language model

from typing import Iterator, Literal, Protocol

import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .tokenizer import CharacterTokenizer

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
## Neural Network Model
#############################################################

class TensorBiGramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(TensorBiGramModel, self).__init__()

        self.vocab_size = vocab_size

        # this will be trained into the bi-gram counting matrix
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 1), int tensor between 0 and vocab_size - 1
        embed = self.embedding(x) # (batch_size, 1, vocab_size)
        
        bi_gram_logits = embed.view(embed.size(0), -1) # (batch_size, vocab_size)

        # returning the logits, it is the log(count_of_next_char_given_current_char) in the counting model
        return bi_gram_logits


class MLPLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, look_back: int, hidden_sizes: list[int]):
        super(MLPLanguageModel, self).__init__()

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
        layers.append(nn.Linear(output_layer_input_size, self.vocab_size))

        self.layers = layers
        self.layers_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, look_back), int tensor between 0 and vocab_size - 1
        embed = self.embedding(x) # (batch_size, look_back, embed_size)

        # flatten
        nn_input = embed.view(embed.size(0), -1) # (batch_size, look_back * embed_size)
        nn_output_logits = self.layers_stack(nn_input) # (batch_size, vocab_size)

        return nn_output_logits

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_state_size: int):
        super(RNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_state_size = hidden_state_size

        self.linear = nn.Linear(input_size + hidden_state_size, hidden_state_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_size)
        # h: (batch_size, hidden_state_size)
        combined = torch.cat([x, h], dim=1) # (batch_size, input_size + hidden_state_size)
        pre_activation = self.linear(combined) # (batch_size, hidden_state_size)
        hidden = torch.tanh(pre_activation) # (batch_size, hidden_state_size)
        return hidden

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_state_size: int):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_state_size = hidden_state_size

        self.linear_z = nn.Linear(input_size + hidden_state_size, hidden_state_size)
        self.linear_r = nn.Linear(input_size + hidden_state_size, hidden_state_size)
        self.linear_h = nn.Linear(input_size + hidden_state_size, hidden_state_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_size)
        # h: (batch_size, hidden_state_size)
        combined = torch.cat([x, h], dim=1) # (batch_size, input_size + hidden_state_size)
        z = torch.sigmoid(self.linear_z(combined)) # (batch_size, hidden_state_size)
        r = torch.sigmoid(self.linear_r(combined)) # (batch_size, hidden_state_size)
        h_masked = r * h # (batch_size, hidden_state_size)
        combined_masked = torch.cat([x, h_masked], dim=1) # (batch_size, input_size + hidden_state_size)
        h_hat = torch.tanh(self.linear_h(combined_masked)) # (batch_size, hidden_state_size)
        hidden = (1 - z) * h + z * h_hat # (batch_size, hidden_state_size)
        return hidden


class MyGRU(nn.Module):
    """Single-layer GRU that mirrors nn.GRU using explicit GRUCell steps."""

    def __init__(self, input_size: int, hidden_state_size: int):
        super().__init__()

        self.input_size = input_size # D
        self.hidden_state_size = hidden_state_size # H

        self.gru_cell = GRUCell(input_size, hidden_state_size)
        self.init_hidden = nn.Parameter(torch.zeros(1, hidden_state_size))

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        if h0 is None:
            hidden = self.init_hidden.expand(batch_size, -1)
        else:
            if h0.dim() == 3:
                hidden = h0[-1]
            else:
                hidden = h0

        outputs = []
        for t in range(seq_len):
            hidden = self.gru_cell(x[:, t, :], hidden)
            outputs.append(hidden)

        output = torch.stack(outputs, dim=1)
        hidden_final = hidden.unsqueeze(0)

        return output, hidden_final


class RNNProtocol(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def step(self, x: torch.Tensor, hidden_prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        ...
    
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...

    @property
    def vocab_size(self) -> int:
        ...
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...
    

class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_state_size: int, cell_type: Literal["rnn", "gru"]):
        super(RNNModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_state_size = hidden_state_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        if cell_type == 'gru':
            self.rnn_cell = GRUCell(embed_size, hidden_state_size)
        else:
            self.rnn_cell = RNNCell(embed_size, hidden_state_size)
        self.output_layer = nn.Linear(hidden_state_size, vocab_size)

        # register the hidden state as a parameter
        self.init_hidden = nn.Parameter(torch.randn(1, hidden_state_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_series_length), int tensor between 0 and vocab_size - 1
        embed = self.embed(x) # (batch_size, max_series_length, embed_size)

        batch_size = embed.size(0)
        max_series_length = embed.size(1)

        # initialize hidden state
        hidden = self.init_hidden.expand(batch_size, -1) # (batch_size, hidden_state_size)
        
        # iterate over the series for each time step to get the hidden states
        hidden_state_series = []
        for t in range(max_series_length):
            embed_t = embed[:, t, :] # (batch_size, embed_size)
            hidden = self.rnn_cell(embed_t, hidden) # (batch_size, hidden_state_size)
            hidden_state_series.append(hidden)
        
        # convert the hidden states to logits
        hidden_states = torch.stack(hidden_state_series, dim=1) # (batch_size, max_series_length, hidden_state_size)
        output_logits = self.output_layer(hidden_states) # (batch_size, max_series_length, vocab_size)

        return output_logits
        
    def step(self, x: torch.Tensor, hidden_prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size,), int tensor between 0 and vocab_size - 1
        # hidden_prev: (batch_size, hidden_state_size)
        if hidden_prev is None:
            hidden_prev = self.init_hidden.expand(x.size(0), -1)
        embed = self.embed(x) # (batch_size, embed_size)
        hidden = self.rnn_cell(embed, hidden_prev) # (batch_size, hidden_state_size)
        logits = self.output_layer(hidden) # (batch_size, vocab_size)
        return logits, hidden


#############################################################
## Neural Model Training
#############################################################

def prepare_n_gram_dataset(samples: list[str], tokenizer: CharacterTokenizer, look_back: int) -> TensorDataset:
    # convert samples to torch tensors
    X = []
    Y = []
    for sample in samples:
        tokens = tokenizer.encode(sample)
        # pad to the left
        tokens = [tokenizer.pad_token] * max(look_back - 1, 0) + [tokenizer.start_token] + tokens + [tokenizer.end_token]
        for i in range(look_back, len(tokens)):
            X.append(tokens[i-look_back:i])
            Y.append(tokens[i])

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, Y)

    return dataset

def prepare_auto_regressive_dataset(samples: list[str], tokenizer: CharacterTokenizer, max_context_len: int | None = None) -> TensorDataset:
    if max_context_len is None:
        max_context_len = max([len(sample) for sample in samples])

    # convert samples to torch tensors
    X = []
    Y = []
    for sample in samples:
        sample_tokens = tokenizer.encode(sample)
        x_tokens = [tokenizer.start_token] + sample_tokens + [tokenizer.end_token]
        # pad to the right
        x_tokens = x_tokens + [tokenizer.pad_token] * (max_context_len - (len(x_tokens) - 2))
        # print(f"tokens: {tokens}, decode: {tokenizer.decode(tokens)}")
        # Y tokens are shifted right by one
        y_tokens = x_tokens[1:] + [tokenizer.pad_token]
        X.append(x_tokens)
        Y.append(y_tokens)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X, Y)

    return dataset

def train_torch_n_gram_model(model: nn.Module, dataset: TensorDataset, batch_size=32, epochs=1000, learning_rate=0.01):
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


def train_auto_regressive_model[T: RNNProtocol](model: T, dataset: TensorDataset, batch_size=32, epochs=1000, learning_rate=0.01, ignore_token=0) -> T:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_samples = len(dataloader.dataset) # type: ignore
    
    # Initialize model, loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, Y_batch in dataloader: # X_batch: (batch_size, max_series_length)
            batch_size = X_batch.shape[0]

            # Forward pass
            outputs = model(X_batch) # (batch_size, max_series_length, vocab_size)
            # Y_batch: (batch_size, max_series_length)
            loss = criterion(outputs.view(-1, model.vocab_size), Y_batch.view(-1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
    
    return model

@torch.no_grad()
def sample_torch_n_gram_model(model: nn.Module, tokenizer: CharacterTokenizer, look_back: int, n_samples: int = 10, max_length: int = 100) -> list[str]:
    samples: list[str] = []

    for _ in range(n_samples):
        current_tokens = [tokenizer.pad_token] * max(0, look_back - 1) + [tokenizer.start_token]
        generated_tokens = []

        for _ in range(max_length):
            current_tokens_tensor = torch.tensor(current_tokens, dtype=torch.long).unsqueeze(0) # (1, look_back)

            with torch.no_grad():
                logits = model(current_tokens_tensor) # (1, vocab_size)

            # randomly pick one token out based on the logits distribution
            next_token = int(torch.multinomial(torch.softmax(logits[0], dim=0), 1).item())
            
            if next_token == tokenizer.end_token:
                break

            generated_tokens.append(next_token)
            current_tokens = current_tokens[1:] + [next_token]
        
        generated_text = tokenizer.decode(generated_tokens)
        samples.append(generated_text)

    return samples

@torch.no_grad()
def sample_auto_regressive_model(model: RNNProtocol, tokenizer: CharacterTokenizer, n_samples: int = 10, max_length: int = 100) -> list[str]:
    samples: list[str] = []

    with torch.no_grad():
        for _ in range(n_samples):
            # initialize the generated series with the input series
            tokens = [tokenizer.start_token]
            hidden = None

            for t in range(max_length):
                current_token = torch.tensor(tokens[t]).unsqueeze(0) # (1,)
                logits, hidden = model.step(current_token, hidden) # logits: (1, vocab_size), hidden: (1, hidden_state_size)

                probs = torch.softmax(logits[0], dim=0)
                next_token = int(torch.multinomial(probs, 1).item()) 
                # print(probs)
                # print(tokenizer.decode([next_token]))
                # breakpoint() # tokenizer.decode([next_token])

                if next_token == tokenizer.end_token:
                    break
                
                tokens.append(next_token)
            
            # remove the starting token
            tokens = tokens[1:]

            generated_text = tokenizer.decode(tokens)
            samples.append(generated_text)

    return samples
