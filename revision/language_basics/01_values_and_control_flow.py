# 1. classify_price
# None -> "missing", < 0 -> "invalid", == 0 -> "free", > 0 -> "paid". check None first.
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


# ---- try it ----
if __name__ == "__main__":
    # assert (classify_price(None), classify_price(-1.5), classify_price(0), classify_price(9.99)) == ('missing', 'invalid', 'free', 'paid')
    # assert (clamp(5, 0, 3), clamp(-2, 0, 3), clamp(2, 0, 3)) == (3, 0, 2)
    # clamp(2, 3, 1)                                                # raises ValueError
    pass
