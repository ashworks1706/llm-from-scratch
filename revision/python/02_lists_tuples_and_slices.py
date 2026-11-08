# lists own an ordered sequence of references to objects; tuples are immutable.
# accessing list[i] is O(1), while inserting at the front shifts elements.
# slices create new lists; arr[start:stop] excludes stop.
# append adds one object; extend adds each element from an iterable.
# sort() mutates; sorted() creates a new list. never confuse their return values.


# Problem 1: reversed copy.
#
# The function takes items, a list of ints, and returns a new list with the same elements in reverse
# order. The caller's list must not change, and the returned list must be a different object. An
# empty list gives an empty list.
#
# Example: [1, 2, 3] gives [3, 2, 1] and the input is still [1, 2, 3].
def reverse_copy(items: list[int]) -> list[int]:
    # return a reversed copy without modifying the caller's list.
    rev_c = items[::-1]
    return rev_c


# Problem 2: rotate a list to the right.
#
# The function takes items, a list of ints, and k, an int, and returns a new list where every element
# moved k places to the right and the elements that fall off the end wrap around to the front. The
# input list is not modified. k may be zero, larger than the length of the list, or negative. A
# negative k rotates to the left, and k is reduced modulo the length so a full turn changes nothing.
# An empty list has no valid rotation, so the function raises ValueError("list cannot be empty").
#
# Example: rotate_right([1, 2, 3, 4, 5], 2) gives [4, 5, 1, 2, 3]. With k = 7 the result is the same
# because 7 mod 5 is 2. With k = -1 the result is [2, 3, 4, 5, 1].
#
# Aim for O(n) time using slices, with no loop over k.
def rotate_right(items: list[int], k: int) -> list[int]:
    if not items:
        raise ValueError("list cannot be empty")
    k = k % len(items)  
    return items[-k:] + items[:-k]



# Problem 3: neighbouring pairs.
#
# The function takes items, a list of ints, and returns a list of tuples holding every pair of
# adjacent elements in order, so the pair at position i is (items[i], items[i + 1]). A list with
# fewer than two elements has no pairs and gives an empty list. The input is not modified.
#
# Example: [1, 2, 3, 4] gives [(1, 2), (2, 3), (3, 4)]. [7] gives [].
def moving_pairs(items: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(items) - 1):
        pairs.append((items[i], items[i + 1]))

    return pairs
