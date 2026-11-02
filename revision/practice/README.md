# practice — debugging and deliberate recall

## error journal
copy this table after every session; record your own observations rather than generic advice.

| date | language | problem | first bug | root cause | correction | rewrite due |
|---|---|---|---|---|---|---|
| — | python | aggregation | missing key | using `+=` before initialization | `get(key, 0)` | next day |

## two-pass rule
pass 1: write the algorithm from memory and test it. pass 2: after debugging, delete your implementation and rewrite it cleanly without referring to the fixed code. repeat the next day and after seven days.

## checkpoints
- [ ] python: 5 array/hash-map problems from a blank file.
- [ ] typescript: typed API aggregator, with runtime validation and tests.
- [ ] rust: borrowing explanation plus a small cache that passes tests.
- [ ] cuda: correct elementwise kernel and reduction, checked against CPU.

## measure improvement
track time to first correct solution, repeated syntax mistakes, failing tests, time complexity, and whether a second independent implementation works.

## anti-patterns
- looking at an AI solution before writing an attempt.
- memorizing `lambda` or `heapq` without knowing why it helps.
- assuming a test passing proves an invariant.
- benchmarking a GPU kernel without synchronizing or comparing output.
- marking a section complete because the explanation seems familiar.
