import random
import string

import torch
import torch.nn as nn

from .tokenizer import CharacterTokenizer

##############################################
## Sample generation
##############################################

def generate_single_sample(min_length: int, max_length: int, stickiness: int, strict: bool) -> str:
    '''
    Generate a random string with sticky character classes. 
    Possible character classes:
    a: lowercase letter
    A: uppercase letter
    0: digit
    
    max_length: maximum length of the string
    stickiness:
        stickiness = 0: no stickiness, each character is independent
        stickiness = 1: each character class will last at least 2 times (unless the string is ended), e.g., aaAAAAA0
        stickiness = N: each character class will last at least N+1 times

    strict: if true, each character class will last exactly N+1 times and then switch to another class
    '''

    assert 0 < min_length <= max_length
    assert stickiness >= 0

    # first generate the character classes
    char_classes = []
    sticky_count = 0
    for _ in range(max_length):
        # can stop at any time, even before reaching the given stickiness
        # 1/4 chance to stop
        if len(char_classes) >= min_length:
            stop = random.randint(1, 4)
            if stop == 1:
                break

        if len(char_classes) == 0:
            char_classes.append(random.choice("aA0"))
            continue

        prev = char_classes[-1]

        # if still in sticky period, keep the same class
        if sticky_count < stickiness:
            sticky_count += 1    
            char_classes.append(prev)
            continue
        
        # otherwise, randomly choose a new class that is different from the previous one
        if strict:
            next_classes = "aA0".replace(prev, "")
        else:
            next_classes = "aA0"
        next = random.choice(next_classes)
        char_classes.append(next)
        if next == prev:
            sticky_count += 1
        else:
            sticky_count = 0
    
    # Then generate the characters from the classes
    chars = []
    for cc in char_classes:
        if cc == "a":
            c = random.choice(string.ascii_lowercase)
        elif cc == "A":
            c = random.choice(string.ascii_uppercase)
        elif cc == "0":
            c = random.choice(string.digits)
        chars.append(c)

    return "".join(chars)

def generate_samples(n_samples: int, min_length: int,  max_length: int, stickiness: int, strict: bool) -> list[str]:
    samples = [generate_single_sample(min_length, max_length, stickiness, strict) for _ in range(n_samples)]
    return samples

##############################################
## Sample validation
##############################################

def validate_one_sample(stickiness: int, strict: bool, sample: str) -> bool:  
    # convert the sample to character classes
    char_classes = []
    for c in sample:
        if c in string.ascii_lowercase:
            char_classes.append("a")
        elif c in string.ascii_uppercase:
            char_classes.append("A")
        elif c in string.digits:
            char_classes.append("0")
        else:
            return False

    # check the stickiness
    sticky_count = 0
    for i in range(len(char_classes)):
        if i == 0:
            continue
        prev = char_classes[i - 1]
        if char_classes[i] == prev:
            sticky_count += 1
            if strict and sticky_count > stickiness:
                return False
            # print(f"{i}, {sample[i]} ({char_classes[i]}): Sticky+1")
        else:
            if sticky_count < stickiness:
                # print(f"{i}, {sample[i]} ({char_classes[i]}): Failed")
                return False
            # print(f"{i}, {sample[i]} ({char_classes[i]}): reset sticky")
            sticky_count = 0

    return True



##############################################
## Baseline Model
##############################################

class BaselineModel:
    '''Baseline model does not train, only generate samples uniformly with random characters (lowercase, uppercase, digits)'''

    def __init__(self):
        pass

    def train(self, samples: list[str]):
        pass

    def generate(self, n_samples: int, max_length: int) -> list[str]:
        samples = []
        for _ in range(n_samples):
            length = random.randint(1, max_length)
            sample = "".join([random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(length)])
            samples.append(sample)
        return samples


###################################################
## NN Model Specialized for Sticky Distribution
###################################################

class StickyNNModel(nn.Module):
    def __init__(self, max_context_size: int, use_manual_init: tuple[int, bool] | None = None) -> None:
        super().__init__()

        known_stickiness, known_strict = use_manual_init if use_manual_init is not None else (None, None)
        if known_stickiness is not None:
            print(f"Using manual initialization: stickiness={known_stickiness}, strict={known_strict}")

        self.max_context_size = max_context_size

        # 6 types of input:
        # pad
        # start
        # end
        # 0-9
        # A-Z
        # a-z
        # this is the tokenizer order, as sorted(['a','z','A','Z','0','1','9']) -> ['0', '1', '9', 'A', 'Z', 'a', 'z']
        self.vocab_size = 26 + 26 + 10 + 3 # V
        embed_size = 6 # E

        # embedding layer, we use one-hot encoding for each character class
        if known_stickiness is not None:
            embed_pad = torch.tensor([[1, 0, 0, 0, 0, 0]]).float()
            embed_start = torch.tensor([[0, 1, 0, 0, 0, 0]]).float()
            embed_end = torch.tensor([[0, 0, 1, 0, 0, 0]]).float()
            embed_0to9 = torch.tensor([[0, 0, 0, 1, 0, 0]]).float().expand(10, -1)
            embed_A2Z = torch.tensor([[0, 0, 0, 0, 1, 0]]).float().expand(26, -1)
            embed_a2z = torch.tensor([[0, 0, 0, 0, 0, 1]]).float().expand(26, -1)
            embed = torch.cat([embed_pad, embed_start, embed_end, embed_0to9, embed_A2Z, embed_a2z], dim=0)
            embed = embed.log()
        else:
            embed = torch.randn(self.vocab_size, embed_size)
        self.embed = nn.Embedding(self.vocab_size, embed_size, _weight=embed) # shape: (V, E)
    
        if known_stickiness is not None:
            position_mask = torch.zeros(max_context_size, max_context_size)
            for i in range(max_context_size):
                for j in range(max_context_size):
                    if j > i: # do not consider future tokens
                        position_mask[i, j] = 0
                    elif i - j <= known_stickiness: # only consider the last N=known_stickiness tokens
                        position_mask[i, j] = 1
                    else:
                        position_mask[i, j] = 0
        else:
            position_mask = torch.rand(max_context_size, max_context_size)
        # lower triangular matrix, only allow the model to see to the past
        self.position_mask = nn.Parameter(torch.tril(position_mask)) # shape: (max_T, max_T)

        # we will use reset rule when the token score is larger than this threshold
        if known_stickiness is not None:
            reset_threshold = (torch.ones(1) * (known_stickiness + 1 - 0.5)).sqrt() # shape: (1,)
        else:
            reset_threshold = torch.randn(1)
        self.reset_threshold = nn.Parameter(reset_threshold) # shape: (1,)

        # temperature for the reset sigmoid function, lower -> more 0-1, higher -> softer
        self.temperature = nn.Parameter(torch.tensor(0.01).sqrt()) # shape: (1,)

        # the output layer: from current input to the next token class

        # case 1: if we are transitioning to the next class based on the current class 
        if known_stickiness is not None:
            class_transition = torch.tensor([
                [1,0,0,0,0,0], # if current input is pad, output pad
                [0,0,0,1,1,1], # if current input is start, output 0-9, A-Z, a-z with equal probability
                [1,0,0,0,0,0], # if current input is end, output pad
                [0,0,1,1,0,0], # if current input is 0-9, still output 0-9, or end
                [0,0,1,0,1,0], # if current input is A-Z, still output A-Z, or end
                [0,0,1,0,0,1], # if current input is a-z, still output a-z, or end
                ]).float()
        else:
            class_transition = torch.randn(embed_size, embed_size)
        self.class_transition = nn.Parameter(class_transition) # shape: (E, E)
        
        # case 2: if we reset the class, i.e. forget about the current class and start over
        if known_strict is not None:
            if not known_strict:
                # case 2.1: non-strict stickiness
                reset_transition = torch.tensor([
                    [1,0,0,0,0,0], # if current input is pad, output pad
                    [0,0,0,1,1,1], # if current input is start, output 0-9, A-Z, a-z with equal probability
                    [1,0,0,0,0,0], # if current input is end, output pad
                    [0,0,1,1,1,1], # if current input is 0-9, output 0-9, A-Z, a-z with equal probability, or end
                    [0,0,1,1,1,1], # if current input is A-Z, output 0-9, A-Z, a-z with equal probability, or end
                    [0,0,1,1,1,1], # if current input is a-z, output 0-9, A-Z, a-z with equal probability, or end
                    ]).float()
            else:
                # case 2.2: strict stickiness
                reset_transition = torch.tensor([
                    [1,0,0,0,0,0], # if current input is pad, output pad
                    [0,0,0,1,1,1], # if current input is start, output 0-9, A-Z, a-z with equal probability
                    [1,0,0,0,0,0], # if current input is end, output pad
                    [0,0,1,0,1,1], # if current input is 0-9, output A-Z, a-z  with equal probability, or end
                    [0,0,1,1,0,1], # if current input is A-Z, output 0-9, a-z with equal probability, or end
                    [0,0,1,1,1,0], # if current input is a-z, output 0-9, A-Z with equal probability, or end
                    ]).float()
        else:
            reset_transition = torch.randn(embed_size, embed_size)
        self.reset_transition = nn.Parameter(reset_transition) # shape: (E, E)

        # output logits
        if known_stickiness is not None:
            output = torch.tensor([
                [10, -20, -20] + [-20] * 10 + [-20] * 26 + [-20] * 26, # padding
                [-20, 10, -20] + [-20] * 10 + [-20] * 26 + [-20] * 26, # starting
                [-20, -20, 10] + [-20] * 10 + [-20] * 26 + [-20] * 26, # ending
                [-20, -20, -20] + [10] * 10 + [-20] * 26 + [-20] * 26, # 0-9
                [-20, -20, -20] + [-20] * 10 + [10] * 26 + [-20] * 26, # A-Z
                [-20, -20, -20] + [-20] * 10 + [-20] * 26 + [10] * 26, # a-z
            ]).float()
        else:
            output = torch.randn(embed_size, self.vocab_size)
        self.output = nn.Parameter(output) # shape: (E, V)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T), int tensor between 0 and vocab_size - 1
        embed = self.embed(x) # (B, T, E)

        # convert the embeddings to class probabilities
        class_prob = torch.softmax(embed, -1)

        B, T, E = embed.shape

        similarity = class_prob @ class_prob.transpose(-2, -1) # (B, T, E) x (B, E, T) -> (B, T, T)

        # apply the position weights
        weights = torch.tril(self.position_mask[:T,:T]) # (T, T)
        similarity = similarity * weights # (B,T,T)

        # calculate the total simularity score for each token
        score = similarity.sum(dim=-1) # (B, T)

        # if score is larger than threshold, use reset rule, otherwise, use the transition rule
        score_scaled = (score - self.reset_threshold ** 2) / self.temperature ** 2
        reset = torch.sigmoid(score_scaled).unsqueeze(-1) # shape: (B, T, 1)
        
        # predict the next class based on wheter to continue the transition or to reset
        
        transition_class = class_prob @ self.class_transition # shape: (B, T, E)
        reset_class = class_prob @ self.reset_transition # shape: (B, T, E)
        next_class = reset * reset_class + (1 - reset) * transition_class # shape: (B, T, E)

        # output logits
        output_logits = next_class @ self.output # shape: (B, T, V)

        return output_logits 

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
            probs = torch.softmax(logits, dim=-1) # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    @property
    def device(self):
        return self.embed.weight.device

@torch.no_grad()
def sample_from_sticky_nn_model(model: StickyNNModel, tokenizer: CharacterTokenizer, max_new_tokens: int, n_samples: int):
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


class StickyRNNModel(nn.Module):
    def __init__(self, use_manual_init: tuple[int, bool] | None = None) -> None:
        super().__init__()
        
        known_stickiness, known_strict = use_manual_init if use_manual_init is not None else (None, None)
        if known_stickiness is not None:
            print(f"Using manual initialization: stickiness={known_stickiness}, strict={known_strict}")

        # 6 types of input:
        # pad
        # start
        # end
        # 0-9
        # A-Z
        # a-z
        # this is the tokenizer order, as sorted(['a','z','A','Z','0','1','9']) -> ['0', '1', '9', 'A', 'Z', 'a', 'z']
        self.vocab_size = 26 + 26 + 10 + 3 # V
        embed_size = 6 # E

        # embedding layer, we use one-hot encoding for each character class
        if known_stickiness is not None:
            embed_pad = torch.tensor([[1, 0, 0, 0, 0, 0]]).float()
            embed_start = torch.tensor([[0, 1, 0, 0, 0, 0]]).float()
            embed_end = torch.tensor([[0, 0, 1, 0, 0, 0]]).float()
            embed_0to9 = torch.tensor([[0, 0, 0, 1, 0, 0]]).float().expand(10, -1)
            embed_A2Z = torch.tensor([[0, 0, 0, 0, 1, 0]]).float().expand(26, -1)
            embed_a2z = torch.tensor([[0, 0, 0, 0, 0, 1]]).float().expand(26, -1)
            embed = torch.cat([embed_pad, embed_start, embed_end, embed_0to9, embed_A2Z, embed_a2z], dim=0)
        else:
            embed = torch.randn(self.vocab_size, embed_size)
        self.embed = nn.Embedding(self.vocab_size, embed_size, _weight=embed) # shape: (V, E)
        
        # the hidden state keep a copy of the previous token, plus a accumulator of how many times we have seen the same kind of token
        if known_stickiness is not None:
            hidden_init = torch.tensor([[1,1,1,1,1,1] + [-1]]).float() # shape: (1, H), init state will be "compatible" to anything / predicted the first token to be anything, with zero accumulator
        else:
            hidden_init = torch.randn(1, 7)
        self.hidden_init = nn.Parameter(hidden_init) # shape: (1, H), init state will be "compatible" to anything / predicted the first token to be anything, with zero accumulator

        # we will reset the hidden state when the accumulator is larger than this threshold
        if known_stickiness is not None:
            reset_threshold = (torch.ones(1) * (known_stickiness + 1 - 0.5)).sqrt() # shape: (1,)
        else:
            reset_threshold = torch.randn(1)
        self.reset_threshold = nn.Parameter(reset_threshold) # shape: (1,)

        # temperature for the reset sigmoid function, lower -> more 0-1, higher -> softer
        self.temperature = nn.Parameter(torch.tensor(0.01)) # shape: (1,)

        # the output layer: from current input to the next token class

        # case 1: if we transition to the next class based on the current class 
        if known_stickiness is not None:
            class_transition = torch.tensor([
                [1,0,0,0,0,0], # if current input is pad, output pad
                [0,0,0,1,1,1], # if current input is start, output 0-9, A-Z, a-z with equal probability
                [1,0,0,0,0,0], # if current input is end, output pad
                [0,0,1,1,0,0], # if current input is 0-9, still output 0-9, or end
                [0,0,1,0,1,0], # if current input is A-Z, still output A-Z, or end
                [0,0,1,0,0,1], # if current input is a-z, still output a-z, or end
                ]).float()
        else:
            class_transition = torch.randn(embed_size, embed_size)
        self.class_transition = nn.Parameter(class_transition) # shape: (E, E)
        
        # case 2: if we reset the class, i.e. forget about the current class and start over
        if known_strict is not None:
            if not known_strict:
                # case 2.1: non-strict stickiness
                reset_transition = torch.tensor([
                    [1,0,0,0,0,0], # if current input is pad, output pad
                    [0,0,0,1,1,1], # if current input is start, output 0-9, A-Z, a-z with equal probability
                    [1,0,0,0,0,0], # if current input is end, output pad
                    [0,0,1,1,1,1], # if current input is 0-9, output 0-9, A-Z, a-z with equal probability, or end
                    [0,0,1,1,1,1], # if current input is A-Z, output 0-9, A-Z, a-z with equal probability, or end
                    [0,0,1,1,1,1], # if current input is a-z, output 0-9, A-Z, a-z with equal probability, or end
                    ]).float()
            else:
                # case 2.2: strict stickiness
                reset_transition = torch.tensor([
                    [1,0,0,0,0,0], # if current input is pad, output pad
                    [0,0,0,1,1,1], # if current input is start, output 0-9, A-Z, a-z with equal probability
                    [1,0,0,0,0,0], # if current input is end, output pad
                    [0,0,1,0,1,1], # if current input is 0-9, output A-Z, a-z  with equal probability, or end
                    [0,0,1,1,0,1], # if current input is A-Z, output 0-9, a-z with equal probability, or end
                    [0,0,1,1,1,0], # if current input is a-z, output 0-9, A-Z with equal probability, or end
                    ]).float()
        else:
            reset_transition = torch.randn(embed_size, embed_size)
        self.reset_transition = nn.Parameter(reset_transition) # shape: (E, E)

        # output logits
        if known_stickiness is not None:
            output = torch.tensor([
                [10, -20, -20] + [-20] * 10 + [-20] * 26 + [-20] * 26, # padding
                [-20, 10, -20] + [-20] * 10 + [-20] * 26 + [-20] * 26, # starting
                [-20, -20, 10] + [-20] * 10 + [-20] * 26 + [-20] * 26, # ending
                [-20, -20, -20] + [10] * 10 + [-20] * 26 + [-20] * 26, # 0-9
                [-20, -20, -20] + [-20] * 10 + [10] * 26 + [-20] * 26, # A-Z
                [-20, -20, -20] + [-20] * 10 + [-20] * 26 + [10] * 26, # a-z
            ]).float()
        else:
            output = torch.randn(embed_size, self.vocab_size)
        self.output = nn.Parameter(output) # shape: (E, V)

    def rnn_cell(self, x: torch.Tensor, hidden: torch.Tensor):
        # x: (B, E)
        # hidden: (B, H)
        expected_class = hidden[:, :-1] # shape: (B, E)
        prev_accum = hidden[:, -1:] # shape: (B, 1)

        # dot product similarity between current input class and expected class
        similarity = x.unsqueeze(1) @ expected_class.unsqueeze(1).transpose(1,2) # shape: (B, 1, E) x (B, E, 1) -> (B, 1, 1)
        similarity = similarity.squeeze(1) # shape: (B, 1)

        # if similarity <= 0, reset the accumulator, if similarity = 1, add 1 to accumulator
        # when in the middle, do an linear interpolatation
        addition_raw = (1 + prev_accum) * similarity # shape: (B, 1)
        addition = torch.relu(addition_raw)  - prev_accum # shape: (B, 1)
        new_accum = prev_accum + addition # shape: (B, 1)

        # if accumulator is larger than threshold, reset the accumulator and the expected class
        new_accum_scaled = (new_accum - self.reset_threshold ** 2) / self.temperature
        reset = torch.sigmoid(new_accum_scaled) # shape: (B, 1)

        # reset the accumulator if needed
        next_accum = (1 - reset) * new_accum

        # predict the next class based on wheter to continue the transition or to reset
        transition_class = x @ self.class_transition # shape: (B, E)
        reset_class = x @ self.reset_transition # shape: (B, E)
        next_class = reset * reset_class + (1 - reset) * transition_class

        # next hidden state is the concatenation of the next class and the next accumulator
        new_hidden = torch.cat([next_class, next_accum], dim=1) # shape: (B, 7)

        # print(new_hidden)
        # breakpoint()
        return new_hidden
    
    def step(self, x: torch.Tensor, hidden_prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B,), int tensor between 0 and vocab_size - 1
        # hidden_prev: (B, H)
        if hidden_prev is None:
            hidden_prev = self.hidden_init.expand(x.size(0), -1)
        embed = self.embed(x) # (B, E)
        hidden = self.rnn_cell(embed, hidden_prev) # (B, H)
        logits = hidden[:, :-1] @ self.output # (B, E)
        return logits, hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T), int tensor between 0 and vocab_size - 1
        embed = self.embed(x) # (B, T, E)

        B, T, E = embed.shape

        # initialize hidden state
        hidden = self.hidden_init.expand(B, -1) # (B, H)
        
        # iterate over the series for each time step to get the hidden states
        hidden_state_series = []
        for t in range(T):
            x_t = embed[:, t, :] # (B, E)
            hidden = self.rnn_cell(x_t, hidden) # (B, H)
            hidden_state_series.append(hidden)
        
        # convert the hidden states to logits
        hidden_states = torch.stack(hidden_state_series, dim=1) # (B, T, H)
        output_logits = hidden_states[:,:,:-1] @ self.output # (B, T, V)

        return output_logits
