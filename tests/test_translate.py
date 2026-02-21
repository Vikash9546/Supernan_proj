"""
tests/test_translate.py
Unit tests for syllable counting and translation constraint enforcement.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.translate import count_syllables_en, count_syllables_hi


def test_english_syllable_counting():
    """English syllable counts should be approximately correct."""
    cases = [
        ("hello", 2),
        ("world", 1),
        ("beautiful", 3),
        ("a", 1),
        ("synchronization", 5),
        ("the quick brown fox", 4),
    ]
    for text, expected in cases:
        actual = count_syllables_en(text)
        # Allow ±1 tolerance for approximation
        assert abs(actual - expected) <= 1, \
            f"'{text}': expected ~{expected} syllables, got {actual}"
        print(f"  ✓ '{text}' → {actual} syllables (expected ~{expected})")

    print("✓ English syllable counting passes")


def test_hindi_syllable_counting():
    """Hindi Devanagari syllable counts should be approximately correct."""
    cases = [
        ("नमस्ते", 3),      # na-mas-te
        ("भारत", 2),        # bha-rat
        ("कम्प्यूटर", 3),    # kam-pyoo-tar (...approximate)
    ]
    for text, expected in cases:
        actual = count_syllables_hi(text)
        # Allow ±2 tolerance for Devanagari approximation
        assert abs(actual - expected) <= 2, \
            f"'{text}': expected ~{expected} syllables, got {actual}"
        print(f"  ✓ '{text}' → {actual} syllables (expected ~{expected})")

    print("✓ Hindi syllable counting passes")


def test_empty_and_edge_cases():
    """Edge cases should not crash."""
    assert count_syllables_en("") == 0
    assert count_syllables_en("   ") == 0
    assert count_syllables_hi("") == 0
    assert count_syllables_en("x") >= 1  # Single consonant = at least 1
    print("✓ Edge cases handled")


if __name__ == "__main__":
    test_english_syllable_counting()
    test_hindi_syllable_counting()
    test_empty_and_edge_cases()
    print("\n✅ All translation tests passed!")
