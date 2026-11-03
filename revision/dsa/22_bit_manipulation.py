# ints are bit vectors: & intersects, | unions, ^ toggles, << >> scale by powers of 2.
# x & (x - 1) clears the lowest set bit; x & -x isolates it. x ^ x == 0 and x ^ 0 == x, so xor cancels pairs.
# bitmask = a set over a small universe, n <= ~20: subsets are the integers 0 .. 2^n - 1, in O(1) per membership test.
# python ints are unbounded and negatives act as infinite two's complement: mask with & 0xFFFFFFFF to emulate 32 bits.
# explain: why does xor of all elements find the single non-duplicated item, and what fails if two items are unique?
# explain: why do hash tables use a power of two size and a mask instead of a modulo?


def count_set_bits(x: int) -> int:
    # popcount for x >= 0 by clearing the lowest set bit in a loop: O(bits set). compare with bin(x).count('1').
    pass


def single_number(nums: list[int]) -> int:
    # every value appears twice except one. xor everything. O(n) time, O(1) space.
    pass


def is_power_of_two(x: int) -> bool:
    # true for 1, 2, 4, ... ; false for 0 and negatives. one expression with x & (x - 1).
    pass


def reverse_bits_32(x: int) -> int:
    # reverse the 32 bit pattern of x (0 <= x < 2^32) and return the unsigned result. 0 stays 0, 1 becomes 2^31.
    pass


def all_subsets_by_mask(items: list[str]) -> list[list[str]]:
    # enumerate subsets by looping mask over range(1 << n) and testing bit i. same output as backtracking subsets in file 13.
    # state when this beats recursion (fixed small n, dp over masks) and when it does not.
    pass


if __name__ == '__main__':
    # assert count_set_bits(x) == bin(x).count('1') for x in range(1024). assert single_number([4, 1, 2, 1, 2]) == 4.
    # assert reverse_bits_32(reverse_bits_32(x)) == x for 200 random 32 bit x.
    pass
