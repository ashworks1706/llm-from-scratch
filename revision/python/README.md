# python — from values to reliable programs

## why
python hides memory management but it does not hide types, mutability or control flow. most bugs in simple coding interviews come from using the wrong *kind* of value: a list where a number is expected, a dictionary key that does not yet exist, or an in-place method where a returned copy was required.

## track

- [01 — objects, collections and control flow](01_objects_and_collections.md)
- [02 — aggregation, sorting and problems](02_patterns_and_exercises.md)
- next: 03 sliding windows, stacks, binary search and heaps
- next: 04 modules, testing, typing, generators, async and profiling

## quick reference — which tool for which question?

| intent | default Python choice |
|---|---|
| total of numbers | `sum(iterable)` |
| maximum by derived score | `max(items, key=lambda x: ...)` |
| new sorted copy | `sorted(items, key=..., reverse=True)` |
| mutate a list's order | `items.sort()` |
| total by key | `dict.get(key, 0)` or `defaultdict(int)` |
| seen already? | `set` |
| enumerate positions | `enumerate(items)` |
| items satisfying condition | comprehension or `filter` |
| at least one / all | `any(...)` / `all(...)` |
| process smallest repeatedly | `heapq` |

## exit gate
implement two sum, duplicate detection, per-market aggregation and simple order execution independently. tests and complexity notes are required.

## source connections
trace how your [inference reference files](../../inference/reference/) manipulate tensors, shape and memory. distinguish plain python loops from vectorized tensor operations.
