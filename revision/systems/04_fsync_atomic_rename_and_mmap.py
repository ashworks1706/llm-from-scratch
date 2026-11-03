# write() only reaches the kernel page cache; fsync forces it to the device. a crash in between can lose or tear data.
# durable atomic replace: write a temp file in the SAME directory, fsync it, os.replace onto the target
# (atomic on one filesystem), then fsync the directory so the rename itself survives a crash.
# system calls are the user/kernel boundary; os.open/os.read/os.write are thin wrappers, file objects add buffering.
# from piramid: apps/engine/database/src/storage/wal/log.rs has sync_on_write, fsync after every WAL entry.
# explain: what does fsync cost, and what do you lose by batching it (group commit)?
# explain: why must the temp file live in the same directory as the target?

import mmap
import os


def atomic_write(path: str, data: bytes) -> None:
    # all-or-nothing replace of path. readers see old or new content, never partial.
    # clean up the temp file if anything fails. use os.open/os.write/os.fsync/os.replace.
    pass


def append_durable(path: str, line: bytes, sync: bool) -> int:
    # append line + b"\n" with O_APPEND via os.open; fsync only when sync is True.
    # return the file size after the append.
    pass


def count_lines_mmap(path: str) -> int:
    # count b"\n" using mmap, not read(). empty file: mmap raises, so handle it.
    pass


def read_raw(path: str, offset: int, length: int) -> bytes:
    # os.open + os.pread; loop until length bytes or EOF since pread may return short.
    pass


if __name__ == '__main__':
    # assert atomic_write twice leaves the second content and no stray temp files in the dir.
    # time append_durable with sync True vs False over 200 lines and print the ratio.
    pass
