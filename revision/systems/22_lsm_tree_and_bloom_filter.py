# an LSM tree turns random writes into sequential ones: writes go to an in-memory sorted memtable (plus a
# WAL for durability); when full it is flushed as an immutable sorted run (SSTable). reads check the
# memtable, then runs newest to oldest; the first hit wins. deletes write a tombstone. compaction merges
# runs, drops shadowed values and old tombstones. a bloom filter per run says "definitely not here" or
# "maybe here", skipping most disk reads for missing keys; it has false positives, never false negatives.
# the runs are in-memory sorted lists here; no files needed.
# explain: what are read, write and space amplification, and which one does compaction reduce?
# explain: size-tiered vs leveled compaction: which favors writes, which favors reads?

import hashlib


class Bloom:
    def __init__(self, bits: int, hashes: int):
        # derive `hashes` positions from one stable hash (hashlib), not python hash().
        pass

    def add(self, key: str) -> None:
        pass

    def maybe_contains(self, key: str) -> bool:
        pass


class LSM:
    TOMBSTONE = object()

    def __init__(self, memtable_limit: int):
        # memtable dict, runs list newest-first, each run = (sorted [(k, v)], Bloom).
        pass

    def put(self, key: str, value: str) -> None:
        # flush the memtable to a new run when it reaches memtable_limit entries.
        pass

    def delete(self, key: str) -> None:
        # write a tombstone, never remove from older runs directly.
        pass

    def get(self, key: str):
        # memtable first, then runs newest to oldest, bloom check then bisect. tombstone -> None.
        pass

    def compact(self) -> None:
        # merge all runs into one, newest wins per key, drop tombstones (safe: this is the oldest level).
        pass


if __name__ == '__main__':
    # random puts/deletes mirrored in a plain dict: assert get agrees before and after compact().
    # measure the bloom false positive rate on 10k absent keys and assert it is below ~5% at 10 bits/key.
    pass
