# slices copy and exclude stop; append adds one object, extend adds each element.
# sort() mutates and returns None; sorted() returns a new list.


# 1. reverse_copy
# return a new reversed list; input unchanged, result is a different object. [] -> [].
def reverse_copy(items: list[int]) -> list[int]:
    # return a reversed copy without modifying the caller's list.
    rev_c = items[::-1]
    return rev_c


# 2. rotate_right
# return a new list rotated right by k (overflow wraps to the front); input unchanged.
# k may be 0, > len, or negative (negative rotates left); reduce k mod len.
# [] -> ValueError("list cannot be empty").
# O(n) with slices, no loop over k.
def rotate_right(items: list[int], k: int) -> list[int]:
    if not items:
        raise ValueError("list cannot be empty")
    k = k % len(items)  
    return items[-k:] + items[:-k]



# 3. moving_pairs
# return every adjacent pair (items[i], items[i + 1]) in order; < 2 elements -> []. input unchanged.
def moving_pairs(items: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(items) - 1):
        pairs.append((items[i], items[i + 1]))

    return pairs


# ---- try it ----
if __name__ == "__main__":
    # print(reverse_copy([1, 2, 3]))                # [3, 2, 1]
    # print(rotate_right([1, 2, 3, 4, 5], 2))       # [4, 5, 1, 2, 3]
    # print(rotate_right([1, 2, 3, 4, 5], 7))       # [4, 5, 1, 2, 3]
    # print(rotate_right([1, 2, 3, 4, 5], -1))      # [2, 3, 4, 5, 1]
    # print(moving_pairs([1, 2, 3, 4]))             # [(1, 2), (2, 3), (3, 4)]
    # print(moving_pairs([7]))                      # []
    pass
