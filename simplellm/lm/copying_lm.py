import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import shutil
import datetime

from .transformer import TransformerLM, TransformerBlock
from .lm import MyGRU
from . import sticky

##############################################
# Vocabulary / special tokens
##############################################

PAD = 0
BOS = 1
END_IN = 2
END_OUT = 3
SEP = 4 # semantic separator
FIRST_TOKEN = 5  # random tokens start from this id
# Chain-of-thought special tokens
INC = FIRST_TOKEN + 10
DEC = FIRST_TOKEN + 11
STEP = FIRST_TOKEN + 12
START_THINK = FIRST_TOKEN + 13
END_THINK = FIRST_TOKEN + 14

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

    def verify(self, predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Default verifier: exact match against reference.
        Returns a bool tensor of shape [B].
        """
        return (predicted == reference).all(dim=1)

class CountingSampleGenerator(LMSampleGenerator):
    def __init__(self, min_digits: int, max_digits: int, min_count: int, max_count: int):
        assert min_digits > 0, "min_digits must be > 0"
        assert max_digits >= min_digits, "max_digits must be >= min_digits"
        assert min_count > 1, "min_count must be > 1"
        assert max_count >= min_count, "max_count must be >= min_count"
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.min_count = min_count
        self.max_count = max_count
        self.max_count_digits = len(str(self.max_count))
        self.min_start = 0 if self.min_digits == 1 else 10 ** (self.min_digits - 1)
        self.max_number = 10 ** self.max_digits - 1
        assert self.max_number - self.max_count >= self.min_start, "Configuration leaves no space for counting outputs"
        self.max_seq_len = (
            1  # BOS
            + self.max_count_digits  # count digits
            + 1  # SEP between count and start
            + self.max_digits  # start digits
            + 1  # END_IN
            + (self.max_digits * self.max_count)  # digits for outputs
            + (self.max_count - 1)  # separators between outputs
            + 1  # END_OUT
        )

    def generate(self, batch_size: int) -> torch.Tensor:
        samples = torch.full((batch_size, self.max_seq_len), PAD, dtype=torch.long)
        for i in range(batch_size):
            sample = self._generate_single_sample()
            samples[i, :len(sample)] = torch.tensor(sample, dtype=torch.long)
        return samples

    @staticmethod
    def pretty_to_str(sample: torch.Tensor) -> str:
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
    def _parse_numbers(token_seq: torch.Tensor) -> list[int] | None:
        """
        Parse a token sequence consisting of digits (FIRST_TOKEN + d) and SEP tokens into integers.
        Returns None if the sequence is malformed.
        """
        numbers: list[int] = []
        digits: list[int] = []
        for tok in token_seq.tolist():
            if tok == PAD:
                break
            if tok == SEP:
                if len(digits) == 0:
                    return None
                numbers.append(int("".join(str(d) for d in digits)))
                digits = []
                continue
            if tok < FIRST_TOKEN:
                return None
            digit = tok - FIRST_TOKEN
            if digit < 0 or digit > 9:
                return None
            digits.append(digit)
        if len(digits) > 0:
            numbers.append(int("".join(str(d) for d in digits)))
        if len(numbers) == 0:
            return None
        return numbers

    def verify(self, predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        A counting sample is correct if the model continues counting (by +1) after END_IN,
        for exactly the referenced count numbers and stops with END_OUT.
        """
        assert predicted.shape == reference.shape
        B, T = predicted.shape
        device = predicted.device
        results = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            ref_seq = reference[i]
            pred_seq = predicted[i]

            end_in_positions = (ref_seq == END_IN).nonzero(as_tuple=True)[0]
            if len(end_in_positions) == 0:
                continue
            end_in_idx = int(end_in_positions[0].item())

            # parse count and start before END_IN from the reference
            prefix_numbers = self._parse_numbers(ref_seq[1:end_in_idx])  # skip BOS
            if not prefix_numbers or len(prefix_numbers) < 2:
                continue
            count, start_num = prefix_numbers[0], prefix_numbers[-1]
            if count < 1:
                continue

            # locate END_OUT in the prediction
            post_end_in = pred_seq[end_in_idx + 1 :]
            end_out_positions = (post_end_in == END_OUT).nonzero(as_tuple=True)[0]
            if len(end_out_positions) == 0:
                continue
            end_out_idx = int(end_out_positions[0].item())

            # parse the predicted numbers after END_IN up to END_OUT
            predicted_numbers = self._parse_numbers(post_end_in[:end_out_idx])
            if not predicted_numbers:
                continue

            if len(predicted_numbers) != count:
                continue

            expected_numbers = [start_num + j + 1 for j in range(count)]
            # ensure everything after END_OUT is PAD
            trailing = post_end_in[end_out_idx + 1 :]
            if predicted_numbers == expected_numbers and (trailing == PAD).all():
                results[i] = True

        return results

    def _encode_number(self, value: int) -> list[int]:
        return [FIRST_TOKEN + int(d) for d in str(value)]

    def _sample_start(self, count: int) -> int:
        max_start = self.max_number - count
        assert max_start >= self.min_start, "No valid starting point for given count"
        return random.randint(self.min_start, max_start)

    def _generate_single_sample(self) -> list[int]:
        """
        Generate a sample of the form:
        [BOS, COUNT, SEP, START, END_IN, NEXT_1, SEP, ..., NEXT_N, END_OUT]
        """
        count = random.randint(self.min_count, self.max_count)
        start = self._sample_start(count)
        return self._build_sample(count, start)

    def _build_sample(self, count: int, start: int) -> list[int]:
        assert count >= 2, "count must be at least 2"
        assert start >= self.min_start, "start below allowed min_start"
        assert start + count <= self.max_number, "start + count exceeds max_number"

        sample = [BOS]
        sample.extend(self._encode_number(count))
        sample.append(SEP)

        sample.extend(self._encode_number(start))
        sample.append(END_IN)

        current = start
        for idx in range(count):
            current += 1
            sample.extend(self._encode_number(current))
            if idx < count - 1:
                sample.append(SEP)
            else:
                sample.append(END_OUT)

        assert len(sample) <= self.max_seq_len, "Sample longer than configured max_seq_len"
        return sample

    def generate_boundary_samples(self) -> torch.Tensor:
        """
        Generate deterministic boundary cases where increment crosses a digit-length boundary
        (e.g., 9->10, 99->100). Returns tensor shaped [N, max_seq_len].
        """
        samples: list[list[int]] = []
        min_count_for_boundary = max(self.min_count, 2)
        for digits in range(self.min_digits, self.max_digits):
            start = (10 ** digits) - 1
            if start < self.min_start:
                continue
            if start + min_count_for_boundary > self.max_number:
                continue
            count = min_count_for_boundary
            try:
                samples.append(self._build_sample(count, start))
            except AssertionError:
                continue

        if len(samples) == 0:
            return torch.empty((0, self.max_seq_len), dtype=torch.long)

        tensor_samples = torch.full((len(samples), self.max_seq_len), PAD, dtype=torch.long)
        for i, s in enumerate(samples):
            tensor_samples[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return tensor_samples


class CountingCoTSampleGenerator(CountingSampleGenerator):
    """
    Counting sample generator with chain-of-thought tokens:
    - START_THINK / END_THINK wrap the reasoning for each output number.
    - INC / DEC mark digit-level increment/decrement operations.
    - STEP separates reasoning steps.
    """
    def __init__(self, min_digits: int, max_digits: int, min_count: int, max_count: int):
        super().__init__(min_digits, max_digits, min_count, max_count)
        self._enforce_len = False
        # Pre-compute a worst-case length sample to size padding conservatively.
        self.max_seq_len = self._compute_max_seq_len()
        self._enforce_len = True

    @property
    def vocab_size(self):
        # digits (10) + 5 CoT tokens, plus special tokens before FIRST_TOKEN
        return FIRST_TOKEN + 10 + 5

    def pretty_to_str(self, sample: torch.Tensor) -> str:
        token_to_str = {
            PAD: "",
            BOS: "",
            END_IN: "<END_IN>",
            END_OUT: "<END_OUT>",
            SEP: ",",
            INC: "<INC>",
            DEC: "<DEC>",
            STEP: "<STEP>",
            START_THINK: "<START_THINK>",
            END_THINK: "<END_THINK>",
        }
        s = []
        for token in sample:
            token_int = int(token.item())
            if token_int in token_to_str:
                s.append(token_to_str[token_int])
            elif FIRST_TOKEN <= token_int < FIRST_TOKEN + 10:
                s.append(str(token_int - FIRST_TOKEN))
            else:
                s.append(f"?{token_int}")
        return " ".join(s)

    def generate(self, batch_size: int, count: int | None = None, start: int | None = None) -> torch.Tensor:
        seqs: list[list[int]] = []
        for _ in range(batch_size):
            seq = self._generate_single_cot_sample(count, start)
            if self._enforce_len:
                assert len(seq) <= self.max_seq_len, "Generated sequence exceeds configured max_seq_len"
            seqs.append(seq)

        samples = torch.full((batch_size, self.max_seq_len), PAD, dtype=torch.long)
        for i, seq in enumerate(seqs):
            samples[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return samples

    def verify(self, predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Verify that the predicted outputs (numbers after END_THINK blocks) correctly continue counting
        for the referenced count numbers.
        """
        assert predicted.shape == reference.shape
        B, _ = predicted.shape
        device = predicted.device
        results = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            ref_seq = reference[i]
            pred_seq = predicted[i]

            end_in_positions = (ref_seq == END_IN).nonzero(as_tuple=True)[0]
            if len(end_in_positions) == 0:
                continue
            end_in_idx = int(end_in_positions[0].item())

            prefix_numbers = self._parse_numbers(ref_seq[1:end_in_idx])  # [count, start]
            if not prefix_numbers or len(prefix_numbers) < 2:
                continue
            count, start_num = prefix_numbers[0], prefix_numbers[-1]

            expected_numbers = [start_num + j + 1 for j in range(count)]
            predicted_numbers = self._extract_outputs_after_think(pred_seq, end_in_idx)

            if predicted_numbers == expected_numbers:
                results[i] = True

        return results

    # -------------------------------
    # Helpers for CoT construction
    # -------------------------------
    def _digits_from_int(self, value: int) -> list[int]:
        return [int(d) for d in str(value)]

    def _encode_digits(self, digits: list[int]) -> list[int]:
        return [FIRST_TOKEN + d for d in digits]

    def _digits_with_marker(self, value: int, marker: int | None = None, *, position: str = "after", digit_index: int | None = None) -> list[int]:
        """
        Encode a number into digit tokens and optionally insert a marker token
        either before or after a specific digit (default: after the last digit).
        """
        tokens = self._encode_number(value)
        if marker is None:
            return tokens

        assert position in {"before", "after"}, "position must be 'before' or 'after'"
        if len(tokens) == 0:
            return tokens

        idx = len(tokens) - 1 if digit_index is None else max(0, min(len(tokens) - 1, digit_index))
        insert_at = idx if position == "before" else idx + 1
        return tokens[:insert_at] + [marker] + tokens[insert_at:]

    def _build_step_tokens(self, counter_tokens: list[int], number_tokens: list[int]) -> list[int]:
        """
        Compose one thinking step: remaining counter digits, SEP, current output digits.
        Digits themselves are not separated by SEP.
        """
        return list(counter_tokens) + [SEP] + list(number_tokens)

    def _carry_run_length(self, digits: list[int]) -> int:
        run = 0
        for d in reversed(digits):
            if d == 9:
                run += 1
            else:
                break
        return run

    def _needs_carry(self, value: int) -> bool:
        # Incrementing by 1 triggers a carry when any trailing digit is 9
        return value % 10 == 9

    def _build_carry_steps(self, remaining_after: int, prev_digits: list[int]) -> list[list[int]]:
        """
        Build intermediate carry steps turning trailing 9s into 0s one digit at a time.
        Example: 99 -> steps: [9 <INC> 0], [<INC> 0 0]
        """
        run_len = self._carry_run_length(prev_digits)
        if run_len == 0:
            return []

        steps: list[list[int]] = []
        run_start = len(prev_digits) - run_len
        for offset in range(run_len - 1, -1, -1):
            pos = run_start + offset
            prefix = prev_digits[:pos]
            zeros_after = run_len - (offset + 1)
            number_tokens = self._encode_digits(prefix) + [INC, FIRST_TOKEN + 0] + [FIRST_TOKEN + 0] * zeros_after
            steps.append(self._build_step_tokens(self._encode_number(remaining_after), number_tokens))

        return steps

    def _build_thinking_steps(self, count: int, start: int) -> list[list[int]]:
        """
        Build the chain-of-thought steps for generating `count` numbers starting from `start`.
        Each number contributes:
          - an action step with DEC/INC markers
          - an optional carry step if the increment crosses a digit boundary
          - a state step with the updated counter and value
        """
        steps: list[list[int]] = []
        remaining = count
        current = start

        for _ in range(count):
            next_value = current + 1
            remaining_after = remaining - 1

            prev_digits = self._digits_from_int(current)
            next_digits = self._digits_from_int(next_value)

            action_counter = self._digits_with_marker(remaining, DEC)
            action_number = self._digits_with_marker(current, INC)
            steps.append(self._build_step_tokens(action_counter, action_number))

            if self._needs_carry(current):
                steps.extend(self._build_carry_steps(remaining_after, prev_digits))

            state_counter = self._encode_number(remaining_after)
            state_number = self._encode_number(next_value)
            steps.append(self._build_step_tokens(state_counter, state_number))

            current = next_value
            remaining = remaining_after

        return steps

    def _extract_outputs_after_think(self, seq: torch.Tensor, end_in_idx: int) -> list[int]:
        """
        Extract actual output numbers (between END_THINK and SEP/END_OUT) ignoring CoT content.
        """
        outputs: list[int] = []
        in_think = False
        collecting_digits: list[int] = []
        end_out_pos: int | None = None
        for rel_idx, tok in enumerate(seq[end_in_idx + 1 :], start=end_in_idx + 1):
            tok_int = int(tok.item())
            if tok_int == START_THINK:
                in_think = True
                collecting_digits = []
                continue
            if tok_int == END_THINK:
                in_think = False
                collecting_digits = []
                continue
            if in_think:
                continue

            if tok_int == SEP or tok_int == END_OUT:
                if len(collecting_digits) > 0:
                    outputs.append(int("".join(str(d) for d in collecting_digits)))
                    collecting_digits = []
                if tok_int == END_OUT:
                    end_out_pos = rel_idx
                    break
            elif FIRST_TOKEN <= tok_int < FIRST_TOKEN + 10:
                collecting_digits.append(tok_int - FIRST_TOKEN)
            elif tok_int == PAD:
                break

        if end_out_pos is not None:
            trailing = seq[end_out_pos + 1 :]
            if not (trailing == PAD).all():
                return []

        return outputs

    def _generate_single_cot_sample(self, count: int | None = None, start: int | None = None) -> list[int]:
        if count is None:
            count = random.randint(self.min_count, self.max_count)
        if start is None:
            start = self._sample_start(count)

        assert count >= 2, "count must be at least 2"
        assert start >= self.min_start, "start below allowed min_start"
        assert start + count <= self.max_number, "start + count exceeds max_number"

        sample: list[int] = [BOS]
        sample.extend(self._encode_number(count))
        sample.append(SEP)
        sample.extend(self._encode_number(start))
        sample.append(END_IN)

        thinking_steps = self._build_thinking_steps(count, start)
        sample.append(START_THINK)
        for idx, step in enumerate(thinking_steps):
            sample.extend(step)
            if idx < len(thinking_steps) - 1:
                sample.append(STEP)
        sample.append(END_THINK)

        current = start
        for idx in range(count):
            current += 1
            sample.extend(self._encode_number(current))
            if idx < count - 1:
                sample.append(SEP)
            else:
                sample.append(END_OUT)

        if self._enforce_len:
            assert len(sample) <= self.max_seq_len, "Generated sequence exceeds configured max_seq_len"
        return sample

    def _compute_max_seq_len(self) -> int:
        """
        Return a conservative upper bound on sequence length to avoid under-sizing
        positional embeddings. Uses an analytic bound and also samples a few
        representative starts near digit boundaries.
        """
        max_count_digits = len(str(self.max_count))
        tokens_per_step_upper = max_count_digits + self.max_digits + 2  # counter digits + SEP + digits + optional marker
        max_steps_per_number = 2 + self.max_digits  # action + state + one per trailing 9 (worst-case)
        think_tokens_per_number = (max_steps_per_number * tokens_per_step_upper) + (max_steps_per_number - 1)  # STEP separators
        header_tokens = 1 + max_count_digits + 1 + self.max_digits + 1  # BOS + count + SEP + start + END_IN
        output_tokens = (self.max_count * self.max_digits) + (self.max_count - 1) + 1  # outputs + SEP + END_OUT
        analytic_bound = header_tokens + 1 + (self.max_count * think_tokens_per_number) + 1 + output_tokens  # START/END_THINK

        candidates: list[int] = [analytic_bound]
        for count in range(self.min_count, self.max_count + 1):
            start_candidates = {
                self.min_start,
                max(self.min_start, self.max_number - count),
                self.max_number - count
            }
            # include boundary starts near powers of 10 to induce carries
            for d in range(1, self.max_digits + 1):
                for delta in [0, 1]:
                    s = (10 ** d) - count - delta
                    if s < self.min_start:
                        continue
                    if s + count > self.max_number:
                        continue
                    start_candidates.add(s)

            for start in start_candidates:
                seq = self._generate_single_cot_sample(count=count, start=start)
                candidates.append(len(seq))

        return max(candidates) if candidates else self.max_seq_len

    def generate_boundary_samples(self) -> torch.Tensor:
        """
        Generate deterministic boundary cases where increments cross digit-length boundaries
        (e.g., 9->10, 99->100). Returned tensor shape: [N, max_seq_len].
        """
        samples: list[list[int]] = []
        min_count_for_boundary = max(self.min_count, 2)
        for digits in range(self.min_digits, self.max_digits):
            start = (10 ** digits) - 1
            if start < self.min_start:
                continue
            if start + min_count_for_boundary > self.max_number:
                continue
            count = min_count_for_boundary
            try:
                seq = self._generate_single_cot_sample(count=count, start=start)
            except AssertionError:
                continue
            samples.append(seq)

        if len(samples) == 0:
            return torch.empty((0, self.max_seq_len), dtype=torch.long)

        tensor_samples = torch.full((len(samples), self.max_seq_len), PAD, dtype=torch.long)
        for i, s in enumerate(samples):
            tensor_samples[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return tensor_samples



class StickySampleGenerator(LMSampleGenerator):
    def __init__(self, min_length: int, max_length: int, stickiness: int, strict: bool):
        assert 0 < min_length <= max_length, "min_length must be in (0, max_length]"
        assert stickiness >= 0, "stickiness must be >= 0"
        self.min_length = min_length
        self.max_length = max_length
        self.stickiness = stickiness
        self.strict = strict

        # Build tokenizer compatible with global special tokens.
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.char2tok = {c: FIRST_TOKEN + i for i, c in enumerate(chars)}
        self.tok2char = {v: k for k, v in self.char2tok.items()}

        # BOS + max content + END
        self.max_seq_len = 1 + max_length + 1

    def generate(self, batch_size: int) -> torch.Tensor:
        raw_samples = sticky.generate_samples(
            n_samples=batch_size,
            min_length=self.min_length,
            max_length=self.max_length,
            stickiness=self.stickiness,
            strict=self.strict,
        )
        return self.convert_samples_to_lm_input(raw_samples)

    def convert_samples_to_lm_input(self, raw_samples: list[str]) -> torch.Tensor:
        samples = torch.full((len(raw_samples), self.max_seq_len), PAD, dtype=torch.long)
        for i, sample in enumerate(raw_samples):
            token_ids = [BOS] + [self.char2tok[c] for c in sample] + [END_OUT]
            assert len(token_ids) <= self.max_seq_len, "Sample longer than configured max_seq_len"
            samples[i, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        return samples

    def pretty_to_str(self, sample: torch.Tensor) -> str:
        tokens = [t for t in sample.tolist() if t != PAD]
        decoded_tokens = []
        for token in tokens:
            if token == BOS:
                continue
            if token == END_OUT:
                decoded_tokens.append("<END>")
                break
            decoded_tokens.append(self.tok2char.get(token, "?"))
        return "".join(decoded_tokens)

    @property
    def vocab_size(self):
        return FIRST_TOKEN + len(self.char2tok)

    def verify(self, predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Verify that the predicted sequence decodes to a sample satisfying the sticky distribution.
        """
        assert predicted.shape == reference.shape
        B, T = predicted.shape
        results = torch.zeros(B, dtype=torch.bool, device=predicted.device)

        for i in range(B):
            seq = predicted[i].tolist()

            # Find start token (BOS)
            try:
                start_idx = seq.index(BOS)
            except ValueError:
                continue

            # Find end token after start; stop early on PAD
            end_idx = None
            for j in range(start_idx + 1, len(seq)):
                tok = seq[j]
                if tok == END_OUT:
                    end_idx = j
                    break
                if tok == PAD:
                    # hit padding before an end token -> invalid
                    end_idx = None
                    break
            if end_idx is None:
                continue

            content_tokens = seq[start_idx + 1 : end_idx]
            if not (self.min_length <= len(content_tokens) <= self.max_length):
                continue

            # ensure trailing tokens are PAD only
            if any(tok != PAD for tok in seq[end_idx + 1 :]):
                continue

            try:
                content = "".join(self.tok2char[tok] for tok in content_tokens)
            except KeyError:
                continue

            if sticky.validate_one_sample(self.stickiness, self.strict, content):
                results[i] = True

        return results



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
    end_in_mask = (lm_seq == END_IN)
    end_in_pos = torch.full((B,), -1, dtype=torch.long, device=device)
    found = end_in_mask.any(dim=1)
    if found.any():
        end_in_pos[found] = end_in_mask[found].float().argmax(dim=1)

    outputs = torch.full((B, max_len), PAD, device=device, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_len):
        # always seed BOS (or first token) before model generation
        if t == 0:
            assert (lm_seq[:, 0] == BOS).all(), "First token must be BOS"
            outputs[:, t] = lm_seq[:, t].clone()
            continue

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

    task_type = "counting_cot" # "copy" or "take_last" or "counting" or "counting_cot" or "sticky"
    model_type = "ptrGen" # "ptrGen" or "tfPtrGen" or "tf"

    # training
    batch_size = 128
    num_steps = 10000
    lr = 1e-3
    use_cosine_lr_schedule = False
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
        min_digits = 1
        max_digits = 3
        min_count = 2
        max_count = 9
        sample_generator = CountingSampleGenerator(min_digits, max_digits, min_count, max_count)
        vocab_size = sample_generator.vocab_size
        model_max_context_size = sample_generator.max_seq_len
        task_prefix = f"minD{min_digits}_maxD{max_digits}_minC{min_count}_maxC{max_count}"
    elif task_type == "counting_cot":
        min_digits = 1
        max_digits = 3
        min_count = 2
        max_count = 9
        sample_generator = CountingCoTSampleGenerator(min_digits, max_digits, min_count, max_count)
        vocab_size = sample_generator.vocab_size
        model_max_context_size = sample_generator.max_seq_len
        task_prefix = f"cot_minD{min_digits}_maxD{max_digits}_minC{min_count}_maxC{max_count}"
    elif task_type == "sticky":
        min_length = 2
        max_length = 12
        stickiness = 1
        strict = False
        sample_generator = StickySampleGenerator(min_length, max_length, stickiness, strict)
        vocab_size = sample_generator.vocab_size
        model_max_context_size = sample_generator.max_seq_len
        task_prefix = f"minL{min_length}_maxL{max_length}_sticky{stickiness}_{'strict' if strict else 'loose'}"
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
    # Cosine LR decay from lr -> 0 over total steps (accounts for resumed max_step)
    start_step = max_step
    total_steps = num_steps
    def cosine_lambda(step_idx: int):
        if not use_cosine_lr_schedule:
            return 1.0
        progress = min(start_step + step_idx, total_steps)
        return 0.5 * (1 + math.cos(math.pi * progress / total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_lambda)
    # use NLLLoss  when model output is probabilities, use CrossEntropyLoss when model output is logits
    criterion = nn.NLLLoss(ignore_index=PAD)

    # validation set
    test_lm_input = sample_generator.generate(batch_size * 10)

    # add boundary samples for counting tasks (e.g. 99->100)
    if task_type == "counting" or task_type == "counting_cot":
        assert isinstance(sample_generator, CountingSampleGenerator)
        boundary_samples = sample_generator.generate_boundary_samples()
        if boundary_samples.numel() > 0:
            test_lm_input = torch.cat([test_lm_input, boundary_samples], dim=0)

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
        scheduler.step()

        # if validation_acc > 0.95 and not lr_stepped_down:
        #     for param_group in opt.param_groups:
        #         param_group['lr'] = lr * 0.1
        #     lr_stepped_down = True
        #     print("Stepped down learning rate to", param_group['lr'])

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                pred = greedy_decode(model, test_lm_input)

                # show 3 good samples
                correct = sample_generator.verify(pred, test_lm_input)
                good_idx = correct.nonzero(as_tuple=True)[0]
                for i in range(min(3, len(good_idx))):
                    print(f"Good Sample {i}:")
                    print("Sample:", sample_generator.pretty_to_str(test_lm_input[good_idx[i],:]))
                    print("Output:", sample_generator.pretty_to_str(pred[good_idx[i],:]))
                    print()

                # show 3 bad samples:
                bad = correct.logical_not()
                bad_idx = bad.nonzero(as_tuple=True)[0]
                for i in range(min(3, len(bad_idx))):
                    print(f"Bad Sample {i}:")
                    print("Sample:", sample_generator.pretty_to_str(test_lm_input[bad_idx[i],:]))
                    print("Output:", sample_generator.pretty_to_str(pred[bad_idx[i],:]))
                    print()

                # print test accuracy
                n_correct = correct.sum().item()
                n_total = correct.size(0)
                validation_acc = n_correct / n_total
                print(f"Validation Accuracy: Prev Max = {max_validation_acc:.4f}, Curr = {validation_acc:.4f} ({n_correct}/{n_total})")
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
                        f.write("TrainStartTime\tStep\tLoss\tMinLoss\tNCorrect\tNTotal\tValAcc\tMaxValAcc\n")
                with open(perf_path, "a") as f:
                    f.write(f"{training_timestamp}\t{step}\t{loss.item():.4f}\t{min_loss:.4f}\t{n_correct}\t{n_total}\t{validation_acc:.4f}\t{max_validation_acc:.4f}\n")

            print("------------------------------------")
            print("")

# python -m simplellm.lm.copying_lm
if __name__ == "__main__":
    train()
