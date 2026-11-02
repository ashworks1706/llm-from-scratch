# 02 — aggregation, sorting and algorithm choice

## problem → structure → operations

before touching code, write what changes as each record is processed. many interview tasks reduce to a small number of state machines.

| question | state | operation |
|---|---|---|
| count each key | dictionary | increment |
| second encounter | seen set | membership then add |
| maximum derived value | current best | compare |
| cheapest available orders | sorted asks | consume capacity |
| moving-window aggregate | running sum and pointers | add incoming, remove outgoing |

## walk through the order matching problem

asks: price 0.60 quantity 10; price 0.65 quantity 20; price 0.70 quantity 50. buy 25.

- sort cheapest first.
- buy min(25,10)=10 at 0.60; remaining 15; cost 6.
- buy min(15,20)=15 at 0.65; remaining 0; total 15.75.
- don't alter caller's input; if total liquidity is insufficient return None.

the invariant is: **remaining quantity + purchased quantity = requested quantity**. total cost is the sum over quantities actually purchased.

## exercises — no solutions in this file

### A. most active market (10 min)
```python
def most_active_market(trades: list[dict]) -> str | None:
    ...
```
return the market with the largest aggregate quantity, not the largest single trade. empty → None; ties → any maximum.

### B. two sum (10 min)
```python
def two_sum(nums: list[int], target: int) -> list[int]:
    ...
```
return indices of two distinct entries. exactly one solution. expected O(n). think about complements and what to store.

### C. execute market buy (15 min)
```python
def execute_market_buy(asks: list[dict], quantity: int) -> float | None:
    ...
```
nonnegative prices/quantities; quantity > 0. sort cheapest available first without modifying `asks`. insufficient liquidity → None.

### D. volume-weighted average price (10 min)
```python
def vwap(trades: list[dict]) -> float | None:
    ...
```
formula: sum(price × quantity) / sum(quantity). zero total quantity → None.

### E. duplicates in detection order (10 min)
```python
def duplicate_ids(ids: list[str]) -> list[str]:
    ...
```
example `["a","b","a","b","a"]` → `["a","b"]`. emit an ID only on its first repeated occurrence.

## test checklist

- [ ] single item / empty list.
- [ ] duplicate keys / all equal / negative values where allowed.
- [ ] insufficient liquidity.
- [ ] original list unchanged.
- [ ] expected time and auxiliary memory complexity.

## what to measure
don't mark the exercise complete when it merely runs. record: time-to-first-working-version, number of bugs, failed tests, and whether you can rewrite it tomorrow.
