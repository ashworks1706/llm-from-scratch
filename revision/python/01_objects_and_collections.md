# 01 — objects, collections and control flow

## why is `sum(trades)` different from `sum(trade["quantity"] for trade in trades)`?

because `sum` consumes **numbers**, not records. a list of dictionaries is a list of records, so we must first extract numeric fields.

```python
trades = [
    {"market": "A", "quantity": 5},
    {"market": "B", "quantity": 12},
    {"market": "A", "quantity": 8},
]
quantities = [t["quantity"] for t in trades]
total = sum(quantities)                    # 25
same_total = sum(t["quantity"] for t in trades)
```

the comprehension constructs a list; the generator expression feeds values one at a time. for a sum, prefer the generator when no list is otherwise needed.

## what is a dictionary, really?

a dictionary maps **hashable keys** to values. a record like `{"market":"A", "quantity":5}` is itself a dictionary, not a valid key. you probably meant the string `"A"`.

```python
totals = {}
for t in trades:
    market = t["market"]
    totals[market] = totals.get(market, 0) + t["quantity"]
# {"A": 13, "B": 12}
```

trace the state:

| iteration | market | previous | add | resulting totals |
|---|---|---:|---:|---|
| 1 | A | 0 | 5 | A:5 |
| 2 | B | 0 | 12 | A:5, B:12 |
| 3 | A | 5 | 8 | A:13, B:12 |

why `.get(key, 0)`? because `totals[key] += amount` tries to **read** the key before writing it, which raises `KeyError` on the first occurrence.

## collections — choose deliberately

- `list`: ordered, mutable, permits repeats. `append` and `pop` at the end are amortized O(1).
- `tuple`: ordered and immutable; useful for fixed records and as a key if its elements are hashable.
- `set`: unique hashable members, average O(1) membership.
- `dict`: maps unique hashable keys to values, average O(1) lookup/update.

## mutable vs returned copy

```python
values = [3, 1, 2]
copy_sorted = sorted(values)      # [1, 2, 3]; values unchanged
values.sort(reverse=True)        # values becomes [3, 2, 1]; returns None
```

why does this distinction matter? a function should not accidentally mutate data owned by its caller. that is especially important with shared order books or cached results.

## common traps

```python
count += 1                 # python has no ++ operator
flag = True                # capital T
result.append(item)        # method call, not append[item]
for i, x in enumerate(xs): # index and value
    ...
```

avoid mutable default parameters:

```python
def add_item(item, output=None):
    if output is None:
        output = []
    output.append(item)
    return output
```

## implement without the reference

1. `count_by_market(trades)` returns a quantity per market.
2. `unique_in_order(values)` removes duplicates without changing first-appearance order.
3. `sort_orders(orders)` returns a new list sorted by price ascending and quantity descending on ties.
4. explain why `if x in a_list` is usually O(n) and `if x in a_set` is average O(1).

write tests for empty inputs, duplicates, negative amounts and missing fields; decide explicitly whether missing keys should raise or be tolerated.

## recall
- what does `dict.items()` return?
- what is the difference between `sort` and `sorted`?
- why can't we use `trade` (a dict) as a dict key?
