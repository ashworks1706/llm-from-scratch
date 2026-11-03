# a singly linked list gives O(1) insert after a known node but O(n) access by index.
# almost every list problem is pointer rewiring; a dummy head node removes the head special case.
# fast and slow pointers find the middle and detect cycles without extra memory.
# draw the pointers before coding: the bug is nearly always losing next before overwriting it.
# python has no pointer arithmetic, so a node is a small class with val and next.
# explain: why does Floyd's cycle detection meet inside the cycle, and why is it O(n) and not O(n * cycle)?
# explain: when is a linked list better than a python list in practice, and when is it never?


class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode | None' = None):
        self.val = val
        self.next = next


def reverse_list(head: ListNode | None) -> ListNode | None:
    # iterative, O(n) time, O(1) space. empty and single-node lists return unchanged.
    # invariant: prev is the reversed prefix, head is the untouched suffix.
    pass


def merge_two_sorted(a: ListNode | None, b: ListNode | None) -> ListNode | None:
    # splice nodes of two sorted lists into one sorted list, reusing nodes. O(n + m).
    # use a dummy head and a tail pointer. one list may be empty. keep it stable: ties take from a.
    pass


def has_cycle(head: ListNode | None) -> bool:
    # slow moves 1, fast moves 2. O(n) time, O(1) space. guard fast and fast.next.
    pass


def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    # one pass: advance fast n steps ahead, then move both until fast reaches the tail.
    # a dummy head handles removing the first node. assume 1 <= n <= length.
    pass


def reorder_list(head: ListNode | None) -> None:
    # in place: L0 -> Ln -> L1 -> Ln-1 -> ... find the middle, reverse the second half, interleave.
    # O(n) time, O(1) space. lengths 0, 1, 2 and odd versus even lengths are the edge cases.
    pass


if __name__ == '__main__':
    # write build(list) -> head and dump(head) -> list helpers, then assert reverse_list(build([1,2,3])) dumps [3,2,1].
    # build a cycle by hand and assert has_cycle; assert remove_nth_from_end on a one-node list gives None.
    pass
