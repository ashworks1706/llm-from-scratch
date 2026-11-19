# 1. classify_price
# None -> "missing", < 0 -> "invalid", == 0 -> "free", > 0 -> "paid". check None first.
# ex: None -> "missing", -1.5 -> "invalid", 0 -> "free", 9.99 -> "paid"
# flat if chain, one early return per case.
def classify_price(price: float | None) -> str:
    # None -> "missing"
    if price is None:
        return "missing"
    # negative -> "invalid"
    if price < 0:
        return "invalid"
    # zero -> "free"
    if price == 0:
        return "free"
    # positive -> "paid"
    return "paid"


# 2. clamp
# return value limited to [low, high]; bounds inclusive, inside -> unchanged.
# low > high -> ValueError("low cannot exceed high"), checked first even if value is in range.
# ex: (5, 0, 3) -> 3, (-2, 0, 3) -> 0, (2, 0, 3) -> 2, (2, 3, 1) -> ValueError
def clamp(value: float, low: float, high: float) -> float:
    # 1. Check whether the range is invalid.
    if low > high:
      raise ValueError("low cannot exceed high") 
    # 2. Check whether value is below low.
    if value < low:
      value = low
    # 3. Check whether value is above high.
    if value > high:
      value = high
    # 4. Otherwise, return the original value.
    return value
