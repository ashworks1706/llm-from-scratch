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
