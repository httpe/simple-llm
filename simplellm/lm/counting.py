# Counting Sample Generation for LMs

import random

def generate_single_counting_sample(max_digits: int, max_count: int):
    '''
    Generate samples like "1,2,3,4,5"
    '''
    assert max_digits > 0
    assert max_count > 1
    
    start = random.randint(0, 10**max_digits - 1)
    count = random.randint(2, max_count)
    end = start + (count-1)
    if end >= 10**max_digits:
        end = 10**max_digits - 1
    return ",".join(str(i) for i in range(start, end + 1))


def generate_samples(n_samples: int, max_digits: int, max_count: int) -> list[str]:
    samples = [generate_single_counting_sample(max_digits, max_count) for _ in range(n_samples)]
    return samples

def validate_one_sample(sample: str) -> bool:
    parts = sample.split(",")
    if len(parts) < 2:
        return False
    for i in range(1, len(parts)):
        try:
            if int(parts[i]) != int(parts[i-1]) + 1:
                return False
        except:
            return False
    return True