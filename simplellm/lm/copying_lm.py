import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .transformer import TransformerLM, TransformerBlock

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

def max_allowed_separator(seq_len):
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

def generate_take_last_samples(batch_size, max_seq_len, vocab_size, min_separators: int | None = None, max_separators: int | None = None):
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
        max_separators = max_allowed_separator(max_content_len)
    assert max_allowed_separator(max_content_len) >= max_separators, "max_separators is more than that allowed by seq_len"
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
        max_sep_for_content = min(max_separators, max_allowed_separator(content_len))
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
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

        # Attention (self-attention)
        self.W_q = nn.Linear(d_model, d_model, bias=False) # query (from GRU state)
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # key (from GRU states)

        # Generator projection: takes [GRU state + context]
        self.gen_proj = nn.Linear(d_model * 2, vocab_size)

        # Gate now depends on [GRU state + context]
        self.gate = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        B, S = x.size()

        # Embed input
        pos_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        x_emb = self.token_embed(x) + self.pos_embed(pos_ids)
        
        # Process with GRU. The output is already causal.
        gru_out, _ = self.gru(x_emb)  # [B, S, D]

        # Causal Self-Attention
        Q = self.W_q(gru_out)               # [B, S, D]
        K = self.W_k(gru_out)               # [B, S, D]

        # Apply RoPE to Q and K
        sin, cos = build_rope_sin_cos(S, self.D, device=x.device) # [S, D]
        Q = apply_rotary_pos_emb(Q, sin, cos)
        K = apply_rotary_pos_emb(K, sin, cos)

        attn_scores = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5)

        # Apply causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        
        pos_probs = F.softmax(attn_scores, dim=-1)      # [B, S, S]

        # Copy distribution for each token is the sum of pos_probs for all of its occurrences in the source history
        x_one_hot = F.one_hot(x, num_classes=self.vocab_size).float()  # [B, S, V]
        P_copy = torch.bmm(pos_probs, x_one_hot)       # [B, S, V]

        # Compute attention-weighted context vector
        context = torch.bmm(pos_probs, gru_out)         # [B, S, D]

        # Combine GRU state with context for generation and gating
        combined = torch.cat([gru_out, context], dim=-1)  # [B, S, 2*D]

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
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

        assert num_layers == 1, "Currently only support 1 transformer layer"
        self.tf_block = TransformerBlock(max_context_size=max_len, embed_size=d_model, n_heads=n_head, head_size=None, ff_hidden_size=None)

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
        transformer_out = self.tf_block(gru_out)  # [B, S, D]

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

def convert_to_lm_input_output(src, tgt_out, pad_token=PAD):
    B, T = src.size()
    device = src.device

    lm_input = torch.full((B, T + tgt_out.shape[1]), pad_token, dtype=torch.long, device=device)
    lm_target = torch.full((B, T + tgt_out.shape[1]), pad_token, dtype=torch.long, device=device)

    for i in range(B):
        # Find where src ends (at END_IN) and tgt_out ends (at END_OUT)
        src_len = (src[i] != pad_token).sum().item()
        tgt_out_len = (tgt_out[i] != pad_token).sum().item()

        # Concatenate src and the content of tgt_out (tgt_out thus no BOS after END_IN)
        # [BOS, ..., END_IN, ..., END_OUT]
        full_seq = torch.cat([src[i, :src_len], tgt_out[i, :tgt_out_len]])

        # Input is the full sequence, target is shifted
        lm_input[i, :len(full_seq)] = full_seq
        lm_target[i, :len(full_seq)-1] = full_seq[1:]

    return lm_input, lm_target

def train():
    vocab_size = 30
    d_model = 16
    max_seq_len = 24
    max_separators = 2
    batch_size = 128
    num_steps = 10000
    lr = 1e-3
    model_type = "copy" # "copy" or "take_last"

    n_head = 1          # Number of attention heads
    num_layers = 1      # Number of transformer layers (THIS IS THE DEPTH)

    # model_save_dir = f"./ptrGen_{model_type}_v{vocab_size}_d{d_model}_l{max_seq_len}"
    model_save_dir = f"./tfBlock4Content_tfHead{n_head}_tfLayer{num_layers}_contentAndAbsPosAttn_{model_type}_v{vocab_size}_d{d_model}_l{max_seq_len}"

    use_saved_model = True
    debug = False

    # fix seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    def gen_samples(n):
        if model_type == "copy":
            return generate_copy_samples(n, max_seq_len, vocab_size)
        else:
            return generate_take_last_samples(n, max_seq_len, vocab_size,  max_separators=max_separators)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # The model's max_len needs to accommodate the longest possible sequence (src + tgt)
    model_max_context_size = max_seq_len * 2

    # possible models:
    # model = PointerGeneratorLM(vocab_size, d_model, max_len=model_max_context_size).to(device)
    model = TransformerPointerGeneratorLM(vocab_size, d_model, max_len=model_max_context_size, n_head=n_head, num_layers=num_layers).to(device)
    # model = TransformerLM(vocab_size, d_model, max_context_size=model_max_context_size, n_heads=n_head, n_layer=num_layers, head_size=None, ff_hidden_size=None).to(device)

    # print model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter Count: {num_params}")

    # find max step model
    max_step = 0
    if use_saved_model:
        saved_models = os.listdir(model_save_dir) if os.path.exists(model_save_dir) else []
        path = None
        for fn in saved_models:
            if fn.endswith(".pt"):
                parts = fn.split("step")
                step = int(parts[1].split(".")[0])
                if step > max_step:
                    max_step = step
                    path = os.path.join(model_save_dir, fn)
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=device))
            print("Loaded saved model from", path)



    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    test_src, _, test_tgt_out = gen_samples(batch_size * 10)
    test_src = test_src.to(device)
    test_tgt_out = test_tgt_out.to(device)
    test_lm_input, _ = convert_to_lm_input_output(test_src, test_tgt_out, pad_token=PAD)

    # debug model for any bad samples
    if debug:
        model.eval()
        with torch.no_grad():
            debug_model(model, test_lm_input)
        model.train()

    for step in range(max_step + 1, num_steps+1):
        src, _, tgt_out = gen_samples(batch_size)
        src, tgt_out = src.to(device), tgt_out.to(device)

        lm_input, lm_target = convert_to_lm_input_output(src, tgt_out, pad_token=PAD)

        P_final = model(lm_input)   # [B, T, V]

        loss = criterion(P_final.reshape(-1, vocab_size), lm_target.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        # # decrease lr after half the steps
        # if step > num_steps // 2:
        #     for param_group in opt.param_groups:
        #         param_group['lr'] = lr * 0.1

        if step % 200 == 0:
            print(f"Step {step}, Loss={loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                pred = greedy_decode(model, test_lm_input)

                # # show 3 good samples
                good = (pred == test_lm_input).all(dim=1)
                good_idx = good.nonzero(as_tuple=True)[0]
                for i in range(min(3, len(good_idx))):
                    print(f"Good Sample {i}:")
                    print("Sample:", test_lm_input[good_idx[i],:])
                    print("Output:", pred[good_idx[i],:])
                    print()

                # show 3 bad samples:
                bad = (pred != test_lm_input).any(dim=1)
                bad_idx = bad.nonzero(as_tuple=True)[0]
                for i in range(min(3, len(bad_idx))):
                    print(f"Bad Sample {i}:")
                    print("Sample:", test_lm_input[bad_idx[i],:])
                    print("Output:", pred[bad_idx[i],:])
                    print()

                # print test accuracy
                acc = good.float().mean().item()
                print(f"Test Accuracy: {acc:.4f}")

                print("------------------------------------")
                print("")

            model.train()

            # save model
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            acc_floor = int(acc * 100)
            path = os.path.join(model_save_dir, f"acc_{acc_floor}_step{step}.pt")
            torch.save(model.state_dict(), path)
            print("Saved model to", path)

if __name__ == "__main__":
    train()