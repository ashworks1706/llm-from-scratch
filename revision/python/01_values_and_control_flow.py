# Problem 1: classify a price.
#
# A product feed gives each item a price that may be missing, invalid or zero. The function takes
# price, a float or None, and returns one of four strings. None gives "missing". A negative price
# gives "invalid". Exactly zero gives "free". Any positive price gives "paid". Check None first so
# the comparisons never see it.
#
# Example: None gives "missing", -1.5 gives "invalid", 0 gives "free", 9.99 gives "paid".
#
# Aim for a flat if chain with one early return per case.
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


# Problem 2: clamp a value into a range.
#
# The function takes value, low and high, all floats, and returns value limited to the closed range
# from low to high. A value below low returns low, a value above high returns high, and a value
# already inside the range, including exactly low or high, is returned unchanged. If low is greater
# than high the range is invalid and the function raises ValueError with the message
# "low cannot exceed high". The range is validated first, even when value would be inside it.
#
# Example: clamp(5, 0, 3) gives 3, clamp(-2, 0, 3) gives 0, clamp(2, 0, 3) gives 2, and
# clamp(2, 3, 1) raises ValueError.
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
