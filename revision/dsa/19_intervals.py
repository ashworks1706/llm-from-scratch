# interval problems are sort-then-sweep. sort by start (or by end for selection problems), then one linear pass.
# merge rule: if next.start <= current.end they overlap (decide up front whether touching [1,2],[2,3] counts), extend the end with max.
# the end of the merged interval is a max, never just the latest end: [1,10],[2,3] stays [1,10].
# for "how many at once" use an event sweep: +1 at each start, -1 at each end, sort events, track the running count.
# cost is O(n log n) for the sort; the pass is O(n). a min heap of end times replaces the sort of events when rooms are asked.
# explain: why must the end of a half open interval sort before the start of another at the same time in the sweep?
# explain: for removing the fewest intervals so none overlap, why sort by end and not by start?

import heapq


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    # merge overlapping closed intervals; touching ones ([1,2],[2,3]) merge. do not mutate the input.
    # empty input returns []. unsorted input is allowed. invariant: the last output interval is the open one.
    pass


def insert_interval(intervals: list[list[int]], new: list[int]) -> list[list[int]]:
    # intervals are sorted and disjoint. insert and merge in O(n) without re-sorting: copy those ending before new,
    # merge those overlapping, copy those starting after.
    pass


def can_attend_all(meetings: list[tuple[int, int]]) -> bool:
    # half open [start, end): a meeting ending at 10 and one starting at 10 do not clash. empty list is True.
    # sort by start, then compare each start with the previous end.
    pass


def min_meeting_rooms(meetings: list[tuple[int, int]]) -> int:
    # fewest rooms so no two overlapping meetings share one (half open). O(n log n).
    # sort by start; a min heap holds the end time of each busy room; reuse a room when heap[0] <= start.
    # the answer is the peak heap size. cross-check against the event sweep version.
    pass


def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    # fewest intervals to remove so the rest do not overlap (touching is fine).
    # sort by end, keep an interval when its start >= last kept end, count the rest as removed.
    pass


if __name__ == '__main__':
    # assert merge_intervals([[1,3],[8,10],[2,6]]) == [[1,6],[8,10]] and merge_intervals([[1,4],[2,3]]) == [[1,4]].
    # assert min_meeting_rooms([(0,30),(5,10),(15,20)]) == 2 and min_meeting_rooms([(1,5),(5,8)]) == 1.
    pass
