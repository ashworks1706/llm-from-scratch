# a process owns an address space; threads inside it share memory and each has its own stack.
# cpython has a GIL: only one thread runs python bytecode at a time, so threads do not speed up
# cpu-bound pure python. they do overlap blocking I/O, because the GIL is released while waiting.
# multiprocessing sidesteps the GIL with separate interpreters, paying pickling and startup cost.
# stdlib only: threading, multiprocessing, time.
# explain: why does sum over a shared int from 4 threads still need a lock even with the GIL?
# explain: when is a process pool slower than a single loop for cpu-bound work?

import multiprocessing
import threading
import time


def burn(n: int) -> int:
    # cpu-bound: sum of i*i for i in range(n). pure python, no I/O. return the sum.
    pass


def run_threaded(fn, args: list[int], workers: int) -> list[int]:
    # run fn(arg) for each arg on workers threads, results in input order.
    # a worker must not mutate shared state without a lock. join every thread.
    pass


def run_multiprocess(fn, args: list[int], workers: int) -> list[int]:
    # same contract with a multiprocessing.Pool. fn must be picklable (module level).
    # guard the entry point with the __main__ check or spawn start methods recurse.
    pass


def timed(label: str, thunk) -> float:
    # return wall seconds from time.perf_counter around thunk(); do not print.
    pass


if __name__ == '__main__':
    # time burn x4 sequentially, threaded and multiprocess; also time 4 x time.sleep(0.5)
    # threaded. assert threaded cpu work is not ~4x faster and threaded sleep is.
    pass
