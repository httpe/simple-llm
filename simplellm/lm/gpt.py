# GPT-2 like Language Model

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import CharacterTokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader

# Dimension Notations:
# B: Batch size
# T: Sequence length
# max(T): Maximum sequence length
# C: Embedding dimension
# H: Attention Head size
# V: Vocabulary size
# F: feed forward network hidden layer size

class FeedForward(nn.Module):
    def __init__(self, embed_size: int, ff_hidden_size: int | None):
        super().__init__()

        if ff_hidden_size is None:
            ff_hidden_size = 4 * embed_size # Magic number 4 is from the GPT-2 paper

        self.linear_expand = nn.Linear(embed_size, ff_hidden_size) # (C, F)
        self.linear_shrink = nn.Linear(ff_hidden_size, embed_size) # (F, C)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        x = self.linear_expand(x) # (B, T, F)
        x = F.relu(x) # (B, T, F)
        x = self.linear_shrink(x) # (B, T, C)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, max_context_size: int, embed_size: int, head_size: int, ):
        super().__init__()

        self.embed_size = embed_size # C
        self.head_size = head_size # H

        # pure matrix multiplication, no bias, apparently
        self.q = nn.Linear(embed_size, head_size, bias=False) # query, (C, H)
        self.k = nn.Linear(embed_size, head_size, bias=False) # key, (C, H)
        self.v = nn.Linear(embed_size, head_size, bias=False) # value, (C, H)

        # KEY IDEA: causal masking, only allow the model to attend to the past
        # E.g. tril 3x3 matrix:
        # 1 0 0
        # 1 1 0
        # 1 1 1
        # declare as buffer, as it's not a learnable parameter
        self.tril = nn.Buffer(torch.tril(torch.ones(max_context_size, max_context_size))) # (max(T), max(T)), only lower-left triangular part is 1


    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        B, T, C = x.shape

        q = self.q(x) # (B, T, H)
        k = self.k(x) # (B, T, H)
        v = self.v(x) # (B, T, H)

        weights =  q @ k.transpose(-2, -1) # (B,T,H) @ (B,H,T) -> (B,T,T)
        weights = weights / (self.head_size ** 0.5) # scale to maintain the variance of the output
        
        # mask out the upper triangular part of the matrix
        mask = self.tril[:T,:T] == 0 # (T,T), true for upper triangular part
        # (B,T,T), for each batch b_i, token t_j, the attention to token k is weights[b_i, t_j, k]
        # if k > j, then mask[b_i, t_j, k] = True, so weights[b_i, t_j, k] = -inf, i.e. no attention
        weights = weights.masked_fill(mask, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (B,T,T)

        v = self.v(x) # (B,T,H)
        out = weights @ v # (B,T,T) @ (B,T,H) -> (B,T,H)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, max_context_size: int, embed_size: int, n_heads: int, head_size: int | None):
        super().__init__()

        if head_size is None:
            # effectively splits the embedding dimension into n_heads
            # i.e., we are forcing total(head_size) == embed_size
            # this is based on the paper, theoretically we can have total(head_size) > embed_size
            assert embed_size % n_heads == 0, "embed_size must be divisible by n_heads"
            head_size = embed_size // n_heads

        self.heads = nn.ModuleList([CausalSelfAttention(max_context_size, embed_size, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads*head_size, embed_size) # (H * n_heads, C)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        B, T, C = x.shape

        heads_out = [head(x) for head in self.heads] # [(B, T, H) for each head]
        out = torch.cat(heads_out, dim=-1) # (B, T, H * n_heads)
        out = self.proj(out) # (B,T,C)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, max_context_size: int, embed_size: int, n_heads: int, head_size: int | None, ff_hidden_size: int | None):
        super().__init__()
        self.attention = MultiHeadAttention(max_context_size, embed_size, n_heads, head_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        x = self.norm1(x) # (B, T, C)
        # KEY IDEA: residual/skip connection around the attention block
        x = self.attention(x) + x # (B, T, C)
        x = self.norm2(x) # (B, T, C)
        x = self.feed_forward(x) + x # (B, T, C)
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, max_context_size: int, n_layer: int, n_heads: int, head_size: int | None, ff_hidden_size: int | None):
        super().__init__()
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size) # (V, C)
        self.position_embedding_table = nn.Embedding(max_context_size, embed_size) # (max(T), C)
        
        self.blocks = nn.Sequential(*[TransformerBlock(max_context_size, embed_size, n_heads=n_heads, head_size=head_size, ff_hidden_size=ff_hidden_size) for _ in range(n_layer)])
        
        self.norm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size) # (C, V)

        self.position_idx = nn.Buffer(torch.arange(max_context_size)) # (max(T),)
        self.max_context_size = max_context_size

    def forward(self, tokens: torch.Tensor, targets=None):
        # tokens: (B, T), targets: (B, T), tensor of integers
        B, T = tokens.shape

        token_embed = self.token_embedding_table(tokens) # (B,T,C)

        position = self.position_idx[:T] # (T,)
        pos_embed = self.position_embedding_table(position) # (T,C)
        
        embed = token_embed + pos_embed # (B,T,C)

        x = self.blocks(embed) # (B,T,C)

        x = self.norm(x) # (B,T,C)

        logits = self.lm_head(x) # (B,T,V)

        return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.max_context_size:] # (B, T)
            # get the predictions
            logits = self(idx_cond) # (B, T, V)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    @property
    def device(self):
        return self.lm_head.weight.device

@torch.no_grad()
def sample_from_gpt_model(model: GPTLanguageModel, tokenizer: CharacterTokenizer, max_new_tokens: int, n_samples: int):
    training = model.training
    model.train(False)

    idx = torch.zeros((n_samples, 1), dtype=torch.long).to(model.device)
    idx[:, 0] = tokenizer.start_token

    token_samples = model.generate(idx=idx, max_new_tokens=max_new_tokens)

    generated_samples = []
    for x in token_samples:
        tokens = x.tolist()
        content_tokens = []
        for i in range(1, len(tokens)):
            token = tokens[i]
            if token == tokenizer.start_token or token == tokenizer.pad_token:
                continue
            if token == tokenizer.end_token:
                break
            content_tokens.append(token)
        sample = tokenizer.decode(content_tokens)
        generated_samples.append(sample)

    model.train(training)

    return generated_samples


def train_gpt_model(model: GPTLanguageModel, dataset: TensorDataset, batch_size=32, epochs=1000, learning_rate=0.01, ignore_token=0) -> GPTLanguageModel:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_samples = len(dataloader.dataset) # type: ignore
    
    # Initialize model, loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, Y_batch in dataloader:
            # X_batch: (B, max(T))

            # Forward pass
            logits = model(X_batch) # (B, T, V)

            # Y_batch: (batch_size, max_series_length)
            loss = criterion(logits.view(-1, logits.size(-1)), Y_batch.view(-1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        if (epoch+1) % 10 == 0:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} Epoch [{epoch+1}/{epochs}], Loss: {total_loss/total_samples:.4f}')
    
    return model

