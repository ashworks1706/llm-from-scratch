# python strings are immutable; build a list of parts and join once, never += in a loop.
# compare characters with a 26-slot count list or a Counter; equal counts mean anagram.
# expand around center finds palindromes in O(n^2) with O(1) space: 2n - 1 centers.
# a variable window with a need/have count is the general form of revision/python/07.
# solve revision/python/07 longest_distinct_substring first; min_window builds on it.
# explain: why does a delimiter alone fail for encode/decode, and why does a length prefix not?
# explain: min_window is O(n + m). why does the left pointer never move backward?


def encode(strs: list[str]) -> str:
    # encode a list of arbitrary strings (they may contain any character) into one string.
    # use a length prefix per item, for example 5#hello. the empty list and empty items must round trip.
    pass


def decode(data: str) -> list[str]:
    # inverse of encode. invariant: decode(encode(x)) == x for every list x.
    pass


def longest_palindrome(text: str) -> str:
    # return the longest palindromic substring; any one on ties. empty input returns empty.
    # expand from each center, odd and even lengths. O(n^2) time, O(1) space.
    pass


def valid_palindrome_one_delete(text: str) -> bool:
    # true when text is a palindrome after deleting at most one character.
    # two pointers; on the first mismatch try skipping left or right once. O(n).
    pass


def min_window(s: str, t: str) -> str:
    # shortest substring of s containing every character of t with multiplicity; empty if none.
    # invariant: formed counts how many distinct chars of t meet their required count.
    # expand right until formed is complete, then shrink left while it stays complete.
    pass


if __name__ == '__main__':
    # round trip encode/decode on [], [''], ['#', '3#a'] and a unicode string.
    # assert min_window('ADOBECODEBANC', 'ABC') == 'BANC' and min_window('a', 'aa') == ''.
    pass
