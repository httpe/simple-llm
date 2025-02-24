# Arithmetics Sample Generation for LMs

import random

def generate_single_addition_sample(max_digits: int):
    a = random.randint(0, 10**max_digits - 1)
    b = random.randint(0, 10**max_digits - 1)
    c = a + b
    return f"{a}+{b}={c}"

def generate_samples(n_samples: int, max_digits: int) -> list[str]:
    samples = [generate_single_addition_sample(max_digits) for _ in range(n_samples)]
    return samples

def validate_one_sample(sample: str) -> bool:
    if sample.count("+") != 1 or sample.count("=") != 1:
        return False
    a, b = sample.split("+")
    if "=" not in b:
        return False
    b, c = b.split("=")
    if not a.isdigit() or not b.isdigit() or not c.isdigit():
        return False
    return int(a) + int(b) == int(c)
