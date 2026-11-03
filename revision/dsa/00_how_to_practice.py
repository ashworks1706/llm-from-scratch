# the solve loop. run it out loud on every problem, in this order.
# 1. restate: inputs, outputs, types, and what empty or missing input returns.
# 2. examples: one normal, one tiny, one adversarial. work them by hand first.
# 3. brute force: say it and its cost before optimizing. it becomes the test oracle.
# 4. complexity: read the target off the constraints before picking a structure.
#    n <= 20 allows 2^n, n <= 1e3 allows n^2, n <= 1e5 needs n log n, n >= 1e9 needs log n or math.
# 5. optimize: find the repeated work, then pick the structure that removes it.
# 6. code: write the loop invariant as a comment before writing the loop.
# 7. test edges: empty, one item, duplicates, negatives, all equal, max size, already sorted.
# every later file in revision/dsa assumes this loop. revision/python/09 shows the assert habit.
# explain: why does writing the brute force first make the fast version safer?
# explain: what do you say when you know the pattern but not why it is correct?


def contains_duplicate(nums: list[int]) -> bool:
    # run the loop on this one: brute force is O(n^2) pairs, target is O(n) time, O(n) space.
    # empty and single-item lists return False.
    pass


if __name__ == '__main__':
    # write a nested-loop oracle inline, then assert both agree on 200 random lists
    # of length 0 to 10 drawn from a small range so duplicates actually happen.
    pass
