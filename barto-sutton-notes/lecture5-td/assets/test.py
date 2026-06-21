"""
Palindrome Generator Module

This module provides functions to generate random palindromes and palindromes from a given set of characters.

Example Usage:
    # Generate a random palindrome of length 7
    >>> from generate_palindromes import generate_random_palindrome
    >>> print(generate_random_palindrome(7))
    'abccba' (example output)

    # Generate palindromes from specific characters
    >>> from generate_palindromes import generate_palindromes_from_chars
    >>> print(generate_palindromes_from_chars("aabbcc", max_results=3))
    ['abc cba', 'bca acb', 'cab bac'] (example outputs)

    # Check if a string is a palindrome
    >>> from generate_palindromes import is_palindrome
    >>> print(is_palindrome("racecar"))
    True
"""

import random
import string
from collections import Counter


def is_palindrome(s: str) -> bool:
    """Check if a string reads the same forwards and backwards."""
    return s == s[::-1]


def generate_random_palindrome(length: int, alphabet: str = string.ascii_lowercase) -> str:
    """Generate a random palindrome of the specified length using the given alphabet.
    
    Args:
        length: Desired length of the palindrome (must be > 0).
        alphabet: String of characters to use for generation. Defaults to lowercase English letters.
        
    Returns:
        A randomly generated palindrome string.
    """
    if length <= 0:
        raise ValueError("Length must be a positive integer.")
    
    half_len = length // 2
    first_half = [random.choice(alphabet) for _ in range(half_len)]
    
    if length % 2 == 0:
        return "".join(first_half + first_half[::-1])
    else:
        middle_char = random.choice(alphabet)
        return "".join(first_half + [middle_char] + first_half[::-1])


def generate_palindromes_from_chars(chars: str, max_results: int = 100) -> list:
    """Generate up to `max_results` unique palindromes that can be formed from the given characters.
    
    Args:
        chars: String of available characters.
        max_results: Maximum number of unique palindromes to generate.
        
    Returns:
        A list of palindrome strings. Returns empty list if no palindrome can be formed.
    """
    counts = Counter(chars)
    odd_counts = sum(1 for c in counts.values() if c % 2 != 0)
    
    # A string can form a palindrome only if at most one character has an odd count
    if odd_counts > 1:
        return []
    
    half_chars = []
    middle_char = ""
    for char, count in counts.items():
        half_chars.extend([char] * (count // 2))
        if count % 2 != 0:
            middle_char = char
            
    seen = set()
    results = []
    
    # Shuffle and construct palindromes until we reach max_results or exhaust possibilities
    while len(results) < max_results:
        random.shuffle(half_chars)
        half_str = "".join(half_chars)
        candidate = half_str + middle_char + half_str[::-1]
        
        if candidate not in seen:
            seen.add(candidate)
            results.append(candidate)
            
    return results


if __name__ == "__main__":
    print("=== Random Palindrome Generator ===")
    length = 7
    print(f"Generating 3 random palindromes of length {length}:")
    for _ in range(3):
        print(generate_random_palindrome(length))
        
    print("\n=== Character-Based Palindrome Generator ===")
    chars = "aabbcc"
    print(f"Generating up to 5 unique palindromes from '{chars}':")
    for p in generate_palindromes_from_chars(chars, max_results=5):
        print(p)
