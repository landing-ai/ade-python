import random


def get_random_number(start: int = 0, end: int = 100) -> int:
    """
    Returns a random integer between start and end (inclusive).
    """
    return random.randint(start, end)
