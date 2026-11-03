# bitcask: an append-only log file of records plus an in-memory hash map key -> (offset, length). a write
# appends and updates the map; a read is one map lookup and one seek (one disk read). deletes append a
# tombstone. everything is sequential I/O, but ALL KEYS must fit in memory and range scans are impossible.
# recovery = replay the log to rebuild the map (hint files make this faster). compaction rewrites only
# the live record of each key into a new file.
# record layout suggestion: 4-byte key length, 4-byte value length, key, value; a tombstone uses a
# reserved length. add a crc32 per record to detect a torn tail write after a crash.
# from piramid: apps/engine/database/src/storage/wal/log.rs is a similar append-only log (one json entry
# per line, a monotonic seq number) that is replayed from a given seq at open.
# explain: how does recovery cope with a half-written last record, and how does crc32 help?

import os
import struct
import zlib


class Bitcask:
    def __init__(self, path: str):
        # open (create) the log in append mode and rebuild the index by replaying it.
        pass

    def put(self, key: bytes, value: bytes) -> None:
        # append a record, then set index[key] = (value_offset, value_len). flush so reads see it.
        pass

    def get(self, key: bytes) -> bytes | None:
        # one lookup + one os.pread. None for missing or deleted keys.
        pass

    def delete(self, key: bytes) -> None:
        pass

    def compact(self) -> None:
        # write live keys to a temp file, atomically replace the log (see 04_), rebuild the index.
        pass

    def close(self) -> None:
        pass


if __name__ == '__main__':
    # random ops mirrored in a dict; reopen the file and assert identical contents.
    # truncate the last record mid-way by hand and assert reopen ignores only the torn record.
    pass
