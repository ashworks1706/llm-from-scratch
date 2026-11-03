# virtual memory maps pages to physical frames; touching a non-resident page is a page fault, and when
# all frames are full a replacement policy picks the victim.
# LRU evicts the least recently used page. clock (second chance) approximates it with one reference bit
# per frame and a rotating hand. belady's optimal evicts the page used farthest in the future (baseline).
# related: solve this first, then revisit inference/reference/paged_kv_cache.py and
# inference/systems/23_kv_cache_policy.rs, which page GPU kv memory the same way.
# explain: why does FIFO show belady's anomaly (more frames, more faults) and LRU does not?

from collections import OrderedDict


def lru_faults(refs: list[int], frames: int) -> int:
    # count page faults for the reference string. OrderedDict, O(1) per access.
    # frames == 0 -> every reference faults.
    pass


def clock_faults(refs: list[int], frames: int) -> int:
    # frames array + reference bits + hand. on hit set the bit. on miss advance the hand,
    # clearing set bits, until a frame with bit 0 is found; replace it and set its bit.
    # fill empty frames before evicting anything.
    pass


def optimal_faults(refs: list[int], frames: int) -> int:
    # evict the resident page whose next use is farthest (or never). O(n * frames) is fine.
    pass


if __name__ == '__main__':
    # classic string 7,0,1,2,0,3,0,4,2,3,0,3,2 with 3 frames: work out the fault counts by hand
    # first, then assert. assert optimal <= lru for random strings.
    pass
