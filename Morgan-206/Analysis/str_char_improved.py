"""Expand an isolate strain string into individual mutation patterns.

A strain is written as slash-separated positions, and a position carrying a
mixture has its alternatives appended, so ``M41LV`` means both ``M41L`` and
``M41V``. This returns one pattern per alternative.

    >>> str_char_improved("M41LV/K103N")
    ['M41V', 'M41L', 'K103N']
    >>> str_char_improved("L74L")
    ['WT']

A position whose alternative equals the wild-type residue is not a mutation
and collapses to ``WT``.
"""

from __future__ import annotations


def _uppercase_positions(segment: str) -> list[int]:
    return [i for i, c in enumerate(segment) if "A" <= c <= "Z"]


def _expand_segment(segment: str) -> list[str]:
    """One position; a mixture of n alternatives yields n patterns.

    The trailing block of ``len(upper) - 1`` characters holds the alternatives,
    and each pattern keeps exactly one of them. ``M41LV`` has three uppercase
    letters, so the trailing block is ``LV`` and the patterns come back as
    ``M41V`` then ``M41L``, in that order.
    """
    upper = _uppercase_positions(segment)
    if len(upper) <= 2:
        return [segment]

    n = len(upper) - 1
    tail = range(len(segment) - n, len(segment))
    patterns = []
    for q in range(1, n + 1):
        keep = len(segment) - q
        patterns.append("".join(c for i, c in enumerate(segment)
                                if i not in tail or i == keep))
    return patterns


def str_char_improved(strain: str) -> list[str]:
    """Every individual mutation pattern a strain string encodes."""
    text = str(strain).replace("*", "")

    patterns: list[str] = []
    for segment in text.split("/"):
        patterns.extend(_expand_segment(segment))

    cleaned = []
    for pattern in patterns:
        pattern = pattern.replace("/", "").replace(" ", "")
        cleaned.append("WT" if pattern and pattern[0] == pattern[-1] else pattern)
    return cleaned
