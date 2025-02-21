import random
import string


##############################################
## Sample generation
##############################################

def generate_single_sample(min_length: int, max_length: int, stickiness: int) -> str:
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
        
        # otherwise, randomly choose a new class
        next = random.choice("aA0")
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

def generate_samples(n_samples: int, min_length: int,  max_length: int, stickiness: int) -> list[str]:
    samples = [generate_single_sample(min_length, max_length, stickiness) for _ in range(n_samples)]
    return samples

##############################################
## Sample validation
##############################################

def validate_one_sample(stickiness: int, sample: str) -> bool:  
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

