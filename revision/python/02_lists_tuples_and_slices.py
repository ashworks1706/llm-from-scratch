# lists own an ordered sequence of references to objects; tuples are immutable.
# accessing list[i] is O(1), while inserting at the front shifts elements.
# slices create new lists; arr[start:stop] excludes stop.
# append adds one object; extend adds each element from an iterable.
# sort() mutates; sorted() creates a new list. never confuse their return values.


def reverse_copy(items: list[int]) -> list[int]:
    # return a reversed copy without modifying the caller's list.
    pass


def rotate_right(items: list[int], k: int) -> list[int]:
    # shift elements right by k positions with wraparound.
    # build a new list and preserve the original; also handle an empty list.
    pass


def moving_pairs(items: list[int]) -> list[tuple[int, int]]:
    # produce adjacent pairs: [2, 4, 8] -> [(2, 4), (4, 8)].
    pass
