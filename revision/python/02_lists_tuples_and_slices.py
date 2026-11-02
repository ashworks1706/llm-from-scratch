# lists own an ordered sequence of references to objects; tuples are immutable.
# accessing list[i] is O(1), while inserting at the front shifts elements.
# slices create new lists; arr[start:stop] excludes stop.
# append adds one object; extend adds each element from an iterable.
# sort() mutates; sorted() creates a new list. never confuse their return values.


def reverse_copy(items: list[int]) -> list[int]:
    # return a reversed copy without modifying the caller's list.
    rev_c = items[::-1]
    return rev_c


def rotate_right(items: list[int], k: int) -> list[int]:
    if not items:
        raise ValueError("list cannot be empty")
    k = k % len(items)  
    return items[-k:] + items[:-k]



def moving_pairs(items: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(items) - 1):
        pairs.append((items[i], items[i + 1]))

    return pairs
