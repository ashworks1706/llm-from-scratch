# a stack answers "what is the most recent unresolved thing". LIFO matches nesting and undo.
# a monotonic stack keeps indices whose values are sorted; each index is pushed once and popped once, so O(n).
# pattern: for next greater element, pop while the top is smaller than the current value; the popped ones resolved.
# store indices, not values, when you need distances or widths.
# bracket matching and min stack basics: revision/python/08 valid_brackets. solve it first.
# explain: the inner while loop looks like O(n^2). why is the total work O(n)?
# explain: for largest rectangle, why do sentinel bars at both ends remove the leftover-stack cleanup?


def min_stack_ops(ops: list[tuple[str, int]]) -> list[int]:
    # simulate push, pop, top, get_min on a min stack with O(1) per op; return results of top and get_min in order.
    # keep a second stack (or pairs) of running minimums. duplicates of the minimum must be tracked.
    pass


def daily_temperatures(temps: list[int]) -> list[int]:
    # out[i] = days until a warmer temperature, 0 if none. O(n) time, O(n) space.
    # invariant: stack holds indices of days still waiting, temperatures non-increasing from bottom to top.
    pass


def eval_rpn(tokens: list[str]) -> int:
    # evaluate reverse polish notation with + - * /. division truncates toward zero, not floor.
    # operand order matters for - and /: the second popped is the left operand.
    pass


def largest_rectangle(heights: list[int]) -> int:
    # largest area in a histogram. O(n) with a stack of increasing heights.
    # when a bar pops, its width spans from the new stack top + 1 to the current index - 1.
    pass


def car_fleet(target: int, pos: list[int], speed: list[int]) -> int:
    # count fleets reaching target. sort by position descending, compute arrival times, a slower car ahead absorbs.
    # O(n log n) from the sort, stack or running max after that.
    pass


if __name__ == '__main__':
    # assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0].
    # assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10 and largest_rectangle([]) == 0.
    pass
