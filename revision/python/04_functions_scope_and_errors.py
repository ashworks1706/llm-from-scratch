# Problem 1: safe ratio.
#
# The function takes numerator and denominator, both floats, and returns numerator divided by
# denominator as a float. When the denominator is zero the division is undefined, and the function
# returns None instead of raising. A zero numerator with a nonzero denominator is an ordinary result
# of 0.0. Negative values divide normally.
#
# Example: safe_ratio(6, 3) gives 2.0, safe_ratio(0, 5) gives 0.0, and safe_ratio(1, 0) gives None.
def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator

# Problem 2: normalize an event.
#
# The function takes event, a dict that must contain an "id" key holding a nonempty string and a
# "quantity" key holding a nonnegative int (a bool does not count as an int here). It returns a NEW
# dict with the same keys and values, and the dict it received must not be modified, so changing the
# result never changes the caller's event. If "id" or "quantity" is missing, has the wrong type, the
# id is empty, or the quantity is negative, the function raises ValueError. It never returns None
# for bad input.
#
# Example: {"id": "e1", "quantity": 3} gives an equal dict that is a different object. A quantity
# of -1 raises ValueError, and so does a missing "id".
def normalize_event(event: dict) -> dict:
    
    if "id" not in event:
      raise ValueError("ID not found") 
    
    if "quantity" not in event:
      raise ValueError("Quantity not found") 
    
    if not type(event["id"]) is str:
      raise ValueError("ID is not a string")
    
    if not type(event["quantity"]) is int:
      raise ValueError("Quantity is not an int")
    
    if event["quantity"] < 0:
      raise ValueError("Quantity cannot be 0")
    
    return { "id": event["id"], "quantity": event["quantity"] }
    

    
