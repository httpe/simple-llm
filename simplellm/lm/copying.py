import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
    Src: [BOS, (TOKEN_1, TOKEN_2, ..., SEP) * n, TOKEN_X, ..., TOKEN_N, END_IN]
    Tgt_in: [BOS, TOKEN_X, ..., TOKEN_N]
    Tgt_out: [TOKEN_X, ..., TOKEN_N, END_OUT]
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

class PointerGeneratorSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()

        self.V = vocab_size
        self.D = d_model
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)

        # Encoder: unidirectional GRU
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)

        # Decoder: unidirectional GRU
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)

        # Attention
        self.W_q = nn.Linear(d_model, d_model, bias=False) # query (from decoder state)
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # key (from encoder outputs)

        # Generator projection: takes [decoder state + context]
        self.gen_proj = nn.Linear(d_model * 2, vocab_size)

        # Gate now depends on [decoder state + context]
        self.gate = nn.Linear(d_model * 2, 1)

    def forward(self, src, tgt):
        B, S = src.size()
        _, T = tgt.size()

        # Encode source
        pos_ids = torch.arange(S, device=src.device).unsqueeze(0).expand(B, -1)
        src_emb = self.token_embed(src) + self.pos_embed(pos_ids)
        enc_out, _ = self.encoder(src_emb)  # [B, S, D]

        # Decode previous output (auto-regressive mode) / training output (teacher forcing mode)
        tgt_pos_ids = torch.arange(T, device=src.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.token_embed(tgt) + self.pos_embed(tgt_pos_ids)
        dec_out, _ = self.decoder(tgt_emb)  # [B,T,D]

        # Attention of output onto the inputs
        Q = self.W_q(dec_out)               # [B,T,D]
        K = self.W_k(enc_out)               # [B,S,D]
        attn_scores = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5)
        pos_probs = F.softmax(attn_scores, dim=-1)      # [B,T,S]

        # Copy distribution for each token is the sum of pos_probs for all of its occurrence in the source
        src_one_hot = F.one_hot(src, num_classes=self.V).float()  # [B,S,V]
        P_copy = torch.bmm(pos_probs, src_one_hot)       # [B,T,V]

        # Compute attention-weighted context vector for unidirectional GRU
        context = torch.bmm(pos_probs, enc_out)         # [B,T,D]

        # Combine decoder state with context for generation and gating
        combined = torch.cat([dec_out, context], dim=-1)  # [B,T,2*D]

        # Context-aware generator (non-copy/generate new token) distribution
        P_gen = F.softmax(self.gen_proj(combined), dim=-1)  # [B,T,V]

        # Context-aware gating: copy vs generate 
        gate = torch.sigmoid(self.gate(combined))           # [B,T,1]

        # Final distribution
        P_final = gate * P_gen + (1 - gate) * P_copy

        return P_final   # [B,T,V]

# Same as above but make the encoder bi-directional, the loss converge faster and is more stable
class PointerGeneratorSeq2SeqBiDirectional(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()

        self.V = vocab_size
        self.D = d_model
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)

        # Encoder: bidirectional GRU
        self.encoder = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)

        # Decoder: unidirectional GRU
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)

        # Attention
        self.W_q = nn.Linear(d_model, d_model, bias=False) # query (from decoder state)
        self.W_k = nn.Linear(d_model * 2, d_model, bias=False)  # key (from encoder outputs), d_model*2 for bidirectional encoder output

        # Generator projection: takes [decoder state + forward context + backward context]
        self.gen_proj = nn.Linear(d_model * 3, vocab_size)

        # Gate now depends on [decoder state + forward context + backward context]
        self.gate = nn.Linear(d_model * 3, 1)

    def forward(self, src, tgt):
        B, S = src.size()
        _, T = tgt.size()

        # Encode source
        pos_ids = torch.arange(S, device=src.device).unsqueeze(0).expand(B, -1)
        src_emb = self.token_embed(src) + self.pos_embed(pos_ids)
        enc_out, _ = self.encoder(src_emb)  # [B, S, 2*D]
        enc_out_split = enc_out.view(B, S, 2, self.D)
        enc_out_forward = enc_out_split[:, :, 0, :]
        enc_out_backward = enc_out_split[:, :, 1, :]

        # Decode previous output (auto-regressive mode) / training output (teacher forcing mode)
        tgt_pos_ids = torch.arange(T, device=src.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.token_embed(tgt) + self.pos_embed(tgt_pos_ids)
        dec_out, _ = self.decoder(tgt_emb)  # [B,T,D]

        # Attention of output onto the inputs
        Q = self.W_q(dec_out)               # [B,T,D]
        K = self.W_k(enc_out)               # [B,S,D]
        attn_scores = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5)
        pos_probs = F.softmax(attn_scores, dim=-1)      # [B,T,S]

        # Copy distribution for each token is the sum of pos_probs for all of its occurrence in the source
        src_one_hot = F.one_hot(src, num_classes=self.V).float()  # [B,S,V]
        P_copy = torch.bmm(pos_probs, src_one_hot)       # [B,T,V]

        # Compute attention-weighted context vector for bidirectional GLU
        context_forward = torch.bmm(pos_probs, enc_out_forward)         # [B,T,D]
        context_backward = torch.bmm(pos_probs, enc_out_backward)        # [B,T,D]

        # Combine decoder state with context for generation and gating
        combined = torch.cat([dec_out, context_forward, context_backward], dim=-1)  # [B,T,4*D]

        # Context-aware generator (non-copy/generate new token) distribution
        P_gen = F.softmax(self.gen_proj(combined), dim=-1)  # [B,T,V]

        # Context-aware gating: copy vs generate 
        gate = torch.sigmoid(self.gate(combined))           # [B,T,1]

        # Final distribution
        P_final = gate * P_gen + (1 - gate) * P_copy

        return P_final   # [B,T,V]

# ============================================================
# Greedy Decoding
# ============================================================

def greedy_decode(model, src, max_seq_len):
    B, S = src.size()
    device = src.device
    dec_input = torch.full((B,1), BOS, dtype=torch.long, device=device)
    
    finished = torch.zeros(B, dtype=torch.bool, device=device)  # Track which sequences are done

    outputs = []
    for _ in range(max_seq_len):
        P_final = model(src, dec_input)   # [B,T,V]
        next_token = P_final[:,-1].argmax(dim=-1, keepdim=True)

        # Mark sequences as finished if they generated END_OUT
        existing_finished = finished.clone()
        newly_finished = (next_token.squeeze() == END_OUT)
        finished = finished | newly_finished

        # For previously finished sequences, force PAD token
        next_token[existing_finished] = PAD

        outputs.append(next_token)       

        # If all sequences are finished, we could break early
        if finished.all():
            # Pad remaining positions
            remaining = max_seq_len - len(outputs)
            for _ in range(remaining):
                outputs.append(torch.full((B, 1), PAD, dtype=torch.long, device=device))
            break

        dec_input = torch.cat([dec_input, next_token], dim=1)

    return torch.cat(outputs, dim=1)


# ============================================================
# Training Loop
# ============================================================

def train():
    vocab_size = 30
    d_model = 32
    max_seq_len = 24
    max_separators = 2
    batch_size = 128
    num_steps = 1000
    lr = 1e-3
    model = "take_last" # or "copy"

    # fix seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    def gen_samples(n):
        if model == "copy":
            return generate_copy_samples(n, max_seq_len, vocab_size)
        else:
            return generate_take_last_samples(n, max_seq_len, vocab_size,  max_separators=max_separators)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = PointerGeneratorSeq2SeqBiDirectional(vocab_size, d_model, max_len=max_seq_len + 5).to(device) # add additional context length for meta tokens
    # print model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter Count: {num_params}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    test_src, test_tgt_in, test_tgt_out = gen_samples(128)
    test_src = test_src.to(device)
    test_tgt_in = test_tgt_in.to(device)
    test_tgt_out = test_tgt_out.to(device)

    for step in range(1, num_steps+1):
        src, tgt_in, tgt_out = gen_samples(batch_size)

        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        # i = 9
        # print(src[i,:])
        # print(tgt_in[i,:])
        # print(tgt_out[i,:])
        # import pdb; pdb.set_trace()

        P_final = model(src, tgt_in)   # [B,T,V]
        loss = criterion(P_final.reshape(-1, vocab_size), tgt_out.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss={loss.item():.4f}")
            pred = greedy_decode(model, test_src, max_seq_len=max_seq_len)

            # show 3 good samples
            good = (pred == test_tgt_out).all(dim=1)
            good_idx = good.nonzero(as_tuple=True)[0]
            for i in range(min(3, len(good_idx))):
                print(f"Good Sample {i}:")
                print("Input:", test_src[good_idx[i],:])
                print("Target IN:", test_tgt_in[good_idx[i],:])
                print("Target OUT:", test_tgt_out[good_idx[i],:])
                print("Output:", pred[good_idx[i],:])
                print()

            # show 3 bad samples:
            bad = (pred != test_tgt_out).any(dim=1)
            bad_idx = bad.nonzero(as_tuple=True)[0]
            for i in range(min(3, len(bad_idx))):
                print(f"Bad Sample {i}:")
                print("Input:", test_src[bad_idx[i],:])
                print("Target IN:", test_tgt_in[bad_idx[i],:])
                print("Target OUT:", test_tgt_out[bad_idx[i],:])
                print("Output:", pred[bad_idx[i],:])
                print()

            # print test accuracy
            acc = good.float().mean().item()
            print(f"Test Accuracy: {acc:.4f}")

            print("------------------------------------")
            print("")

if __name__ == "__main__":
    train()