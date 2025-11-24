import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import shutil
import datetime

from .transformer import TransformerLM, TransformerBlock
from .lm import MyGRU

##############################################
# Vocabulary / special tokens
##############################################

PAD = 0
BOS = 1
END_IN = 2
END_OUT = 3
SEP = 4 # semantic separator
FIRST_TOKEN = 5  # random tokens start from this id

##############################################
# Sample generation
##############################################

def convert_seq2seq_samples_to_lm_sample(src, tgt_out, pad_token=PAD):
    B, T = src.size()
    device = src.device

    lm_input = torch.full((B, T + tgt_out.shape[1]), pad_token, dtype=torch.long, device=device)

    for i in range(B):
        # Find where src ends (at END_IN) and tgt_out ends (at END_OUT)
        src_len = (src[i] != pad_token).sum().item()
        tgt_out_len = (tgt_out[i] != pad_token).sum().item()

        # Concatenate src and the content of tgt_out (tgt_out thus no BOS after END_IN)
        # [BOS, ..., END_IN, ..., END_OUT]
        full_seq = torch.cat([src[i, :src_len], tgt_out[i, :tgt_out_len]])

        # Input is the full sequence, target is shifted
        lm_input[i, :len(full_seq)] = full_seq

    return lm_input

class LMSampleGenerator:
    def generate(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError()

    def pretty_to_str(self, sample: torch.Tensor) -> str:
        return str(sample.tolist())
    
    def get_lm_output(self, lm_input: torch.Tensor) -> torch.Tensor:
        '''
        Given lm_input of shape [B, T], return lm_output of shape [B, T]
        '''
        B, T = lm_input.size()
        lm_output = torch.full((B, T), PAD, dtype=torch.long, device=lm_input.device)
        
        lm_output[:, :-1] = lm_input[:, 1:].clone()

        return lm_output

class CountingSampleGenerator(LMSampleGenerator):
    def __init__(self, min_digits: int, max_digits: int, min_count: int, max_count: int):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.min_count = min_count
        self.max_count = max_count

    def generate(self, batch_size: int) -> torch.Tensor:
        samples = self.generate_counting_samples(batch_size, self.min_digits, self.max_digits, self.min_count, self.max_count)
        return samples

    def pretty_to_str(self, sample: torch.Tensor) -> str:
        token_to_str = {
            PAD: "",
            BOS: "",
            END_IN: "<END_IN>",
            END_OUT: "<END_OUT>",
            SEP: ","
        }
        s = []
        for token in sample:
            token_int = int(token.item())
            if token_int in token_to_str:
                s.append(token_to_str[token_int])
            else:
                s.append(str(token_int - FIRST_TOKEN))
        return " ".join(s)

    @property
    def vocab_size(self):
        return FIRST_TOKEN + 10  # digits 0-9

    @staticmethod
    def generate_counting_samples(batch_size: int, min_digits: int, max_digits: int, min_count: int, max_count: int) -> torch.Tensor:
        '''
        Generate counting samples:
        [BOS, "9", SEP, "1", "0", SEP, "1", "1", END_IN, "1", "2", SEP, "1", "3", END_OUT, PAD, ..., PAD]
        '''
        assert min_digits > 0, "min_digits must be > 0"
        assert max_digits > 0, "max_digits must be > 0"
        assert max_digits >= min_digits, "max_digits must be >= min_digits"
        assert min_count > 1, "min_count must be > 1"
        assert max_count > 1, "max_count must be > 1"
        assert max_count >= min_count, "max_count must be >= min_count"

        max_seq_len = 1 + max_digits * max_count + (max_count - 1) + 2  # BOS + n * digits + (n-1) * SEP + END_IN + END_OUT

        samples = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)

        for i in range(batch_size):
            sample = [BOS]
            n_num = random.randint(min_count, max_count)  # count of numbers to generate
            n_input = n_num - 1  # count of numbers as input, only output one more number for now

            for j in range(n_num):
                if j == 0:
                    min_x = 0 if min_digits == 1 else 10 ** (min_digits - 1)
                    x = random.randint(min_x, 10 ** max_digits - n_num)  # ensure we have enough room to count up
                else:
                    x = x + 1
                digits = [int(d) for d in str(x)]
                tokens = [d + FIRST_TOKEN for d in digits]  # shift to token ids
                sample.extend(tokens)
                if j < n_num - 1:
                    if j == n_input - 1:
                        sample.append(END_IN)
                    else:
                        sample.append(SEP)
                else:
                    sample.append(END_OUT)
            if len(sample) > max_seq_len:
                breakpoint()
            samples[i, :len(sample)] = torch.tensor(sample, dtype=torch.long)

        return samples



class CopySampleGenerator(LMSampleGenerator):
    def __init__(self, max_seq_len: int, vocab_size: int):
        assert vocab_size > FIRST_TOKEN, "vocab_size must be > FIRST_TOKEN, so we have enough room for special tokens"
        assert max_seq_len >= 2, "seq_len must be at least 2 to fit BOS and END_IN"
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def generate(self, batch_size: int) -> torch.Tensor:
        src, _, tgt_out = self.generate_copy_samples(batch_size, self.max_seq_len, self.vocab_size)
        lm_input = convert_seq2seq_samples_to_lm_sample(src, tgt_out, pad_token=PAD)
        return lm_input

    @staticmethod
    def generate_copy_samples(batch_size, max_seq_len, vocab_size):
        """
        Generate samples of form:
        Src:        [BOS,       TOKEN 1, TOKEN 2, ..., TOKEN N, END_IN,     PAD 1, ..., PAD M]
        Tgt_in:     [BOS,       TOKEN 1, TOKEN 2, ..., TOKEN N, END_OUT,    PAD 1, ..., PAD M]
        Tgt_out:    [TOKEN 1,   TOKEN 2, TOKEN 3, ..., END_OUT, PAD 1,      PAD 2, ..., PAD M+1]
        """
        assert vocab_size > FIRST_TOKEN, "vocab_size must be > FIRST_TOKEN, so we have enough room for special tokens"
        assert max_seq_len >= 2, "seq_len must be at least 2 to fit BOS and END_IN"

        # max_seq_len == 5
        # max_content_len == 3
        max_content_len = max_seq_len - 2  # exclude BOS and END_IN

        # [5, 9, 7]
        content = torch.randint(FIRST_TOKEN, vocab_size, (batch_size, max_content_len))

        src = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)
        tgt_in = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)
        tgt_out = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)
        for i in range(batch_size):
            # content_len == 2
            content_len = random.randint(1, max_content_len) # make sure the sequence is non-empty (not just BOS and END)

            # [BOS, 5, 9, END_IN, PAD]
            src[i, 0] = BOS
            src[i, 1:content_len + 1] = content[i, :content_len].clone()
            src[i, content_len + 1] = END_IN

            # [BOS, 5, 9, END_OUT, PAD]
            tgt_in[i, :] = src[i, :].clone()
            tgt_in[i, content_len + 1] = END_OUT

        # [5, 9, END_OUT, PAD, PAD]
        tgt_out[:, :-1] = tgt_in[:, 1:].clone()

        return src, tgt_in, tgt_out

class TakeLastSampleGenerator(LMSampleGenerator):
    def __init__(self, max_seq_len: int, vocab_size: int, min_separators: int | None = None, max_separators: int | None = None):
        assert vocab_size > FIRST_TOKEN, "vocab_size must be > FIRST_TOKEN, so we have enough room for special tokens"
        assert max_seq_len >= 2, "seq_len must be at least 2 to fit BOS and END_IN"
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.min_separators = min_separators
        self.max_separators = max_separators

    def generate(self, batch_size: int) -> torch.Tensor:
        src, _, tgt_out = self.generate_take_last_samples(batch_size, self.max_seq_len, self.vocab_size, self.min_separators, self.max_separators)
        lm_input = convert_seq2seq_samples_to_lm_sample(src, tgt_out, pad_token=PAD)
        return lm_input

    @classmethod
    def max_allowed_separator(cls, seq_len):
        # [num, SEP, [num, SEP] * N, num]
        # 1: [num] -> 0
        # 2: [num, num] -> 0
        # 3: [num, SEP, num] -> 1
        # 4: [num, SEP, num, num] -> 1
        # 5: [num, SEP, num, SEP, num] -> 2
        # 6: [num, SEP, num, SEP, num, num] -> 2
        # 7: [num, SEP, num, SEP, num, SEP, num] -> 3
        if seq_len <= 2:
            return 0
        return 1 + (seq_len - 3) // 2

    @classmethod
    def generate_take_last_samples(cls, batch_size, max_seq_len, vocab_size, min_separators: int | None = None, max_separators: int | None = None):
        '''
        Generate samples of form:
        Src:        [BOS, (TOKEN_1, TOKEN_2, ..., SEP) * n, TOKEN_X, ..., TOKEN_N, END_IN,  PAD, ..., PAD]
        Tgt_in:     [BOS, TOKEN_X, ...,     TOKEN_N, PAD, ..., PAD]
        Tgt_out:    [TOKEN_X, ..., TOKEN_N, END_OUT, PAD, ..., PAD]
        '''

        assert vocab_size > FIRST_TOKEN, "vocab_size must be > FIRST_TOKEN, so we have enough room for special tokens"
        assert max_seq_len >= 2, "seq_len must be at least 2 to fit BOS and END_IN"

        # max_seq_len == 7
        # max_content_len == 5
        max_content_len = max_seq_len - 2  # exclude BOS and END_IN

        # max_separator == 2
        if max_separators is None:
            max_separators = cls.max_allowed_separator(max_content_len)
        assert cls.max_allowed_separator(max_content_len) >= max_separators, "max_separators is more than that allowed by seq_len"
        if min_separators is None:
            min_separators = 0
        assert min_separators >= 0 and min_separators <= max_separators

        # content == [7, 6, 5, 8, 9]
        content = torch.randint(FIRST_TOKEN, vocab_size, (batch_size, max_content_len)) # real content expect for BOS and END_IN

        src = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)
        tgt_in = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)
        tgt_out = torch.full((batch_size, max_seq_len), PAD, dtype=torch.long)

        # Insert separators in random places (not consecutive)
        for i in range(batch_size):
            # content_len == 3
            content_len = random.randint(1, max_content_len)

            # num_separators == 1
            max_sep_for_content = min(max_separators, cls.max_allowed_separator(content_len))
            min_sep_for_content = min(min_separators, max_sep_for_content)
            proposed_num_separators = random.randint(min_sep_for_content, max_sep_for_content)  # Random number of separators

            # Choose valid positions (exclude 0 and N-1)
            # valid_positions == [1]
            valid_positions = list(range(1, content_len - 1))

            # To avoid consecutive separators, pick with spacing
            # chosen: {1}
            chosen = set()
            while len(chosen) < proposed_num_separators and len(valid_positions) > 0:
                pos = random.choice(valid_positions)
                chosen.add(pos)
                # ensure no neighbor will be chosen
                valid_positions.remove(pos)
                if pos-1 in valid_positions:
                    valid_positions.remove(pos-1)
                if pos+1 in valid_positions:
                    valid_positions.remove(pos+1)

            # Replace with separator
            # content[i, :]: [7, SEP, 5, 8, 9]
            for pos in chosen:
                content[i, pos] = SEP
            
            # last_sep_pos == 1
            last_sep_pos = max(chosen) if len(chosen) > 0 else -1
            # len_last_part == 1
            len_last_part = int(content_len - last_sep_pos - 1)

            # [BOS, 7, SEP, 5, END_IN, PAD, PAD]
            src[i, 0] = BOS
            src[i, 1:content_len + 1] = content[i, :content_len].clone()
            src[i, content_len + 1] = END_IN

            # [BOS, 5, END_OUT, PAD, PAD, PAD, PAD]
            tgt_in[i, 0] = BOS
            tgt_in[i, 1:len_last_part + 1] = content[i, last_sep_pos+1:last_sep_pos+1+len_last_part].clone()
            tgt_in[i, len_last_part + 1] = END_OUT

        # [5, END_OUT, PAD, PAD, PAD, PAD, PAD]
        tgt_out[:, :-1] = tgt_in[:, 1:].clone()

        return src, tgt_in, tgt_out

##############################################
# NN Model Definition
##############################################

DEBUG_GROUND_TRUTH = None

def rotate_every_two(x):
    # x: [..., D]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return x_rotated.flatten(start_dim=-2)

def apply_rotary_pos_emb(x, sin, cos):
    # x: [B, S, D], sin/cos: [S, D]
    return (x * cos.unsqueeze(0)) + (rotate_every_two(x) * sin.unsqueeze(0))

def build_rope_sin_cos(seq_len, dim, device):
    # dim must be even
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float().unsqueeze(1) # [S,1]
    sinusoid = positions * inv_freq.unsqueeze(0) # [S, D/2]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    # expand to D by interleaving
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1) # [S, D]
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1) # [S, D]
    return sin, cos

class PointerGeneratorLM(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()

        self.vocab_size = vocab_size
        self.D = d_model
        assert self.D % 2 == 0, "d_model must be even for RoPE"
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)

        # Auto-regressive core: unidirectional GRU
        self.gru = MyGRU(d_model, d_model)

        # Attention (self-attention)
        self.W_q = nn.Linear(2 * d_model, 2 * d_model, bias=False) # query (from GRU state and raw input embedding)
        self.W_k = nn.Linear(2 * d_model, 2 * d_model, bias=False)  # key (from GRU states and raw input embedding)
        self.W_qp = nn.Linear(2 * d_model, d_model, bias=False) # query for abs pos (from GRU states and raw input embedding)

        # Generator projection: takes [GRU state + context (prob weighted avg of (GRU state + raw input embedding))]
        self.gen_proj = nn.Linear(d_model * 3, vocab_size)

        # Gate now depends on [GRU state + context (prob weighted avg of (GRU state + raw input embedding))]
        self.gate = nn.Linear(d_model * 3, 1)

    def forward(self, x):
        B, S = x.size()

        # Embed input
        x_emb = self.token_embed(x)
        
        # Process with GRU. The output is already causal.
        gru_out, _ = self.gru(x_emb)  # [B, S, D]

        # Causal Self-Attention
        attention_input = torch.cat([gru_out, x_emb], dim=-1)  # [B, S, 2*D]
        Q = self.W_q(attention_input)               # [B, S, 2*D]
        K = self.W_k(attention_input)               # [B, S, 2*D]

        # Apply RoPE to Q and K
        sin, cos = build_rope_sin_cos(S, 2 * self.D, device=x.device) # [S, 2*D]
        Q = apply_rotary_pos_emb(Q, sin, cos)
        K = apply_rotary_pos_emb(K, sin, cos)

        content_scores = torch.bmm(Q, K.transpose(1,2)) # [B, S, S]

        # absolute position based attention score
        key_abs_pos_ids = torch.arange(S, device=x.device).unsqueeze(0).repeat(B, 1) # [B, S]
        abs_pos_embeddings = self.pos_embed(key_abs_pos_ids) # [B, S, D]
        Q_pos = self.W_qp(attention_input)               # [B, S, D]
        abs_pos_scores = torch.bmm(Q_pos, abs_pos_embeddings.transpose(1, 2))

        # Combine content and absolute position scores for pointer-generation
        pointer_scores = (content_scores + abs_pos_scores) / (Q.size(-1)**0.5)

        # Apply causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        pointer_logits = pointer_scores.masked_fill(mask, -torch.inf)
        pos_probs = F.softmax(pointer_logits, dim=-1) # [B, S, S]

        # Copy distribution for each token is the sum of pos_probs for all of its occurrences in the source history
        x_one_hot = F.one_hot(x, num_classes=self.vocab_size).float()  # [B, S, V]
        P_copy = torch.bmm(pos_probs, x_one_hot)       # [B, S, V]

        # Compute attention-weighted context vector
        context = torch.bmm(pos_probs, attention_input)         # [B, S, 2 * D]

        # Combine GRU state with context for generation and gating
        combined = torch.cat([gru_out, context], dim=-1)  # [B, S, 3*D]

        # Context-aware generator (non-copy/generate new token) distribution
        P_gen = F.softmax(self.gen_proj(combined), dim=-1)  # [B, S, V]

        # Context-aware gating: copy vs generate 
        gate = torch.sigmoid(self.gate(combined))           # [B, S, 1]

        # Final distribution
        P_final = gate * P_gen + (1 - gate) * P_copy

        if DEBUG_GROUND_TRUTH is not None:
            pred_next = P_final[:, -1].argmax(dim=-1) # [B]
            ground_truth_next = DEBUG_GROUND_TRUTH[:, S] # [B]
            if (pred_next != ground_truth_next).any():
                print(f"Debugging the step to generate index {S}")
                breakpoint()

        return P_final   # [B, S, V]


class TransformerPointerGeneratorLM(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_head, num_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.D = d_model
        assert self.D % 2 == 0, "d_model must be even for RoPE"
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.abs_pos_embed = nn.Embedding(max_len, d_model)

        # Auto-regressive core: unidirectional GRU
        self.gru = MyGRU(d_model, d_model)

        self.tf_blocks = nn.Sequential(*[TransformerBlock(max_len, d_model, n_heads=n_head, head_size=None, ff_hidden_size=None) for _ in range(num_layers)])

        # Generator projection: takes [transformer state + attention wavg context]
        self.gen_proj = nn.Linear(d_model * 2, vocab_size)

        # Gate now depends on [transformer state + attention wavg context]
        self.gate = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        B, S = x.size()
        device = x.device

        # Embed input
        x_emb = self.token_embed(x)  # [B, S, D]
        
        # Process with GRU. The output is already causal.
        gru_out, _ = self.gru(x_emb)  # [B, S, D]

        # transformer reasoning block
        transformer_out = self.tf_blocks(gru_out)  # [B, S, D]

        # Use transformer_out as the basis for pointer generation, as both Q and K
        Q_ptr = transformer_out
        K_ptr = transformer_out
        
        # We can still add our robust position information here for the final pointing step
        # sin, cos = build_rope_sin_cos(S, self.D, device=device)
        # Q_ptr_rope = apply_rotary_pos_emb(Q_ptr, sin, cos)
        # K_ptr_rope = apply_rotary_pos_emb(K_ptr, sin, cos)

        # content based self-attention score
        content_scores = torch.bmm(Q_ptr, K_ptr.transpose(1, 2))
        
        # absolute position based attention score
        key_abs_pos_ids = torch.arange(S, device=device).unsqueeze(0).repeat(B, 1) # [B, S]
        abs_pos_embeddings = self.abs_pos_embed(key_abs_pos_ids) # [B, S, D]
        abs_pos_scores = torch.bmm(Q_ptr, abs_pos_embeddings.transpose(1, 2))

        # Combine content and absolute position scores for pointer-generation
        pointer_scores = (content_scores + abs_pos_scores) / (Q_ptr.size(-1)**0.5)

        # Apply the causal mask to the pointer scores as well
        mask_ptr = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        pointer_scores = pointer_scores.masked_fill(mask_ptr, -torch.inf)
        
        pos_probs = F.softmax(pointer_scores, dim=-1) # [B, S, S]

        # Copy distribution for each token is the sum of pos_probs for all of its occurrences in the source history
        x_one_hot = F.one_hot(x, num_classes=self.vocab_size).float()  # [B, S, V]
        P_copy = torch.bmm(pos_probs, x_one_hot)       # [B, S, V]

        # Compute attention-weighted context vector
        context = torch.bmm(pos_probs, transformer_out)         # [B, S, D]

        # Combine transformer state with context for generation and gating
        combined = torch.cat([transformer_out, context], dim=-1)  # [B, S, 2*D]

        # Context-aware generator (non-copy/generate new token) distribution
        P_gen = F.softmax(self.gen_proj(combined), dim=-1)  # [B, S, V]

        # Context-aware gating: copy vs generate 
        gate = torch.sigmoid(self.gate(combined))           # [B, S, 1]

        # Final distribution
        P_final = gate * P_gen + (1 - gate) * P_copy

        if DEBUG_GROUND_TRUTH is not None:
            pred_next = P_final[:, -1].argmax(dim=-1) # [B]
            ground_truth_next = DEBUG_GROUND_TRUTH[:, S] # [B]
            if (pred_next != ground_truth_next).any():
                print(f"Debugging the step to generate index {S}")
                breakpoint()

        return P_final   # [B, S, V]

class WrappedTransformerLM(TransformerLM):
    def forward(self, x):
        # x: [B, S]
        logits = super().forward(x)  # [B, S, V]
        P_final = F.softmax(logits, dim=-1)  # [B, S, V]
        return P_final

# ============================================================
# Greedy Decoding
# ============================================================

def greedy_decode(model, lm_seq):
    B, S = lm_seq.size()
    device = lm_seq.device
    max_len = S

    # compute per-example index of END_IN
    end_in_pos = (lm_seq == END_IN).int().argmax(dim=1)  # [B]

    outputs = torch.full((B, max_len), PAD, device=device, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_len):
        # sequences that should start generating (we copy from src until t > end_in_pos)
        to_generate = (t > end_in_pos)  # [B]  (True: we are in generation phase)
        if not to_generate.any():
            # copy from source for all examples this step
            outputs[:, t] = lm_seq[:, t].clone()
            continue

        input_t = outputs[:, :t]  # [B, t]
        if input_t.size(1) == 0:
            # Model expects some length; feed a 1-step BOS maybe — but your model can handle zero-length?
            # Easiest: feed a single BOS for all (or pad). We'll feed the prefix as-is if t==0 that branch never runs because to_generate false.
            pass

        P_final = model(input_t)   # [B, T, V], where T == t
        next_token = P_final[:, -1].argmax(dim=-1) # [B]

        # For examples not yet in generation phase, copy from source (preserve)
        next_token[~to_generate] = lm_seq[~to_generate, t]

        # mark newly finished
        newly_finished = (next_token == END_OUT)
        # prevent tokens after finished from being non-PAD
        next_token[finished] = PAD

        finished = finished | newly_finished
        outputs[:, t] = next_token.clone()

        if finished.all():
            break

    return outputs


# ============================================================
# Training Loop
# ============================================================

def debug_model(model, test_lm_input):
    global DEBUG_GROUND_TRUTH
    pred = greedy_decode(model, test_lm_input)
    bad = (pred != test_lm_input).any(dim=1)
    bad_idx = bad.nonzero(as_tuple=True)[0]
    if len(bad_idx) == 0:
        return
    print("Debugging model on a bad sample...")
    bad_sample = test_lm_input[bad_idx[0], :].unsqueeze(0)
    print("Bad sample:", bad_sample)
    print("Prediction:", pred[bad_idx[0], :])
    DEBUG_GROUND_TRUTH = bad_sample 
    greedy_decode(model, bad_sample)
    DEBUG_GROUND_TRUTH = None


def train():
    # fix seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    task_type = "copy" # "copy" or "take_last" or "counting"
    model_type = "ptrGen" # "ptrGen" or "tfPtrGen" or "tf"

    # training
    batch_size = 128
    num_steps = 10000
    lr = 1e-3
    use_saved_model = False
    save_model = True
    debug = False
    device = "cpu" # device = "cuda" if torch.cuda.is_available() else "cpu"

    # tasks
    sample_generator: LMSampleGenerator
    if task_type == "copy":
        vocab_size = 15
        max_seq_len = 24
        sample_generator = CopySampleGenerator(max_seq_len, vocab_size)
        model_max_context_size = max_seq_len * 2 # The model's max_len needs to accommodate the longest possible sequence (src + tgt)
        task_prefix = f"maxSeq{max_seq_len}"
    elif task_type == "take_last":
        vocab_size = 15
        max_seq_len = 24
        max_separators = 3
        sample_generator = TakeLastSampleGenerator(max_seq_len, vocab_size, max_separators=max_separators)
        model_max_context_size = max_seq_len * 2 # The model's max_len needs to accommodate the longest possible sequence (src + tgt)
        task_prefix = f"maxSeq{max_seq_len}_maxSep{max_separators}"
    elif task_type == "counting":
        max_seq_len = 24
        min_digits = 1
        max_digits = 3
        min_count = 2
        max_count = 5
        sample_generator = CountingSampleGenerator(min_digits, max_digits, min_count, max_count)
        vocab_size = sample_generator.vocab_size
        model_max_context_size = 1 + max_digits * max_count + (max_count - 1) + 2
        task_prefix = f"minD{min_digits}_maxD{max_digits}_minC{min_count}_maxC{max_count}"
    else:
        raise ValueError("Unknown task_type")
    
    # models
    if model_type == "ptrGen":
        d_model = 32
        model = PointerGeneratorLM(vocab_size, d_model, max_len=model_max_context_size).to(device)
        model_prefix = f"ptrGen_rawInputPassThrough_posAttn_resLayerNormGRU"
    elif model_type == "tfPtrGen":
        d_model = 32
        n_head = 2
        num_layers = 1
        model = TransformerPointerGeneratorLM(vocab_size, d_model, max_len=model_max_context_size, n_head=n_head, num_layers=num_layers).to(device)
        model_prefix = f"tfPtrGen_tfHead{n_head}_tfLayer{num_layers}"
    elif model_type == "tf":
        d_model =  24
        n_head = 2
        num_layers = 3
        model = WrappedTransformerLM(vocab_size, d_model, max_context_size=model_max_context_size, n_heads=n_head, n_layer=num_layers, head_size=None, ff_hidden_size=None).to(device)
        model_prefix = f"tf_tfHead{n_head}_tfLayer{num_layers}"
    else:
        raise ValueError("Unknown model_type")
    
    # print model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter Count: {num_params}")

    model_desc = f"{model_prefix}_vocab{vocab_size}_dModel{d_model}_maxCtx{model_max_context_size}_params{num_params}"
    result_dir = f"./simplellm/lm/saved_models/{task_type}/{task_prefix}/{model_desc}"
    checkpoint_dir = os.path.join(result_dir, "checkpoints")

    if not use_saved_model and save_model:
        if os.path.exists(result_dir):
            # move to backup dir
            backup_dir = result_dir + "_backup_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(result_dir, backup_dir)
            print("Moved existing model dir to backup:", backup_dir)

    # copy this file to model save dir for record
    if save_model:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        training_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        script_name = training_timestamp + "_" + os.path.basename(__file__)
        shutil.copy(__file__, os.path.join(result_dir, script_name))
        lm_script_name = training_timestamp + "_lm.py"
        shutil.copy(os.path.join(os.path.dirname(__file__), "lm.py"), os.path.join(result_dir, lm_script_name))

    # find and load max step model
    max_step = 0
    if use_saved_model:
        saved_models = os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
        path = None
        for fn in saved_models:
            if fn.endswith(".pt"):
                parts = fn.split("step_")
                step = int(parts[1].split(".")[0])
                if step > max_step:
                    max_step = step
                    path = os.path.join(checkpoint_dir, fn)
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=device))
            print("Loaded saved model from", path)

    # optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # use NLLLoss  when model output is probabilities, use CrossEntropyLoss when model output is logits
    criterion = nn.NLLLoss(ignore_index=PAD)

    # validation set
    test_lm_input = sample_generator.generate(batch_size * 10)

    if task_type == "counting":
        has_special_sample = False
        for i in range(len(test_lm_input)):
            s = sample_generator.pretty_to_str(test_lm_input[i])
            input_part = s.split("<END_IN>")[0]
            output_part = s.split("<END_IN>")[1]
            last_input_num = int(input_part.split(',')[-1].replace(' ', ''))
            first_output_num = int(output_part.split(',')[0].replace(' ', '').replace('<END_OUT>', ''))
            if len(str(last_input_num)) != len(str(first_output_num)):
                print("Test Sample with different length output:", s)
                has_special_sample = True
        assert has_special_sample, "Test set should have at least one sample with different length output"

    # debug model for any bad samples
    if debug:
        model.eval()
        with torch.no_grad():
            debug_model(model, test_lm_input)
        model.train()

    # LR scheduler: step down LR by 10x when validation acc > 90%
    # validation_acc = 0.0
    # lr_stepped_down = False

    print("Starting training...")
    print("------------------------------------")

    min_loss = float('inf')
    max_validation_acc = 0.0

    for step in range(max_step + 1, num_steps+1):
        lm_input = sample_generator.generate(batch_size)
        lm_output = sample_generator.get_lm_output(lm_input)

        P_final = model(lm_input)   # [B, T, V]

        # clamp for numerical stability and use NLL on log-probs
        log_p = torch.log(P_final.clamp(min=1e-12))

        loss = criterion(log_p.reshape(-1, vocab_size), lm_output.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        # if validation_acc > 0.95 and not lr_stepped_down:
        #     for param_group in opt.param_groups:
        #         param_group['lr'] = lr * 0.1
        #     lr_stepped_down = True
        #     print("Stepped down learning rate to", param_group['lr'])

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                pred = greedy_decode(model, test_lm_input)

                # # show 3 good samples
                good = (pred == test_lm_input).all(dim=1)
                good_idx = good.nonzero(as_tuple=True)[0]
                for i in range(min(3, len(good_idx))):
                    print(f"Good Sample {i}:")
                    print("Sample:", sample_generator.pretty_to_str(test_lm_input[good_idx[i],:]))
                    print("Output:", sample_generator.pretty_to_str(pred[good_idx[i],:]))
                    print()

                # show 3 bad samples:
                bad = (pred != test_lm_input).any(dim=1)
                bad_idx = bad.nonzero(as_tuple=True)[0]
                for i in range(min(3, len(bad_idx))):
                    print(f"Bad Sample {i}:")
                    print("Sample:", sample_generator.pretty_to_str(test_lm_input[bad_idx[i],:]))
                    print("Output:", sample_generator.pretty_to_str(pred[bad_idx[i],:]))
                    print()

                # print test accuracy
                validation_acc = good.float().mean().item()
                print(f"Validation Accuracy: Prev Max = {max_validation_acc:.4f}, Curr = {validation_acc:.4f}")
                max_validation_acc = max(max_validation_acc, validation_acc)

            model.train()

            print(f"Step {step}, Loss: Prev Min = {min_loss:.4f}, Curr = {loss.item():.4f}")
            min_loss = min(min_loss, loss.item())

            if save_model:
                path = os.path.join(checkpoint_dir, f"{training_timestamp}_step_{step}.pt")
                torch.save(model.state_dict(), path)
                print("Saved model to", path)

                # performance
                perf_path = os.path.join(result_dir, f"training_performance.txt")
                if not os.path.exists(perf_path):
                    with open(perf_path, "w") as f:
                        f.write("TrainStartTime\tStep\tLoss\tMinLoss\tValAcc\tMaxValAcc\n")
                with open(perf_path, "a") as f:
                    f.write(f"{training_timestamp}\t{step}\t{loss.item():.4f}\t{min_loss:.4f}\t{validation_acc:.4f}\t{max_validation_acc:.4f}\n")

            print("------------------------------------")
            print("")

# python -m simplellm.lm.copying_lm
if __name__ == "__main__":
    train()
