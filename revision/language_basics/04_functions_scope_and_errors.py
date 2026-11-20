# 1. safe_ratio
# return numerator / denominator; denominator 0 -> None (don't raise).
# 0 / nonzero -> 0.0, negatives divide normally.
def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator

# 2. normalize_event
# event needs "id": nonempty str and "quantity": int >= 0 (a bool doesn't count as int).
# return a NEW equal dict; don't mutate the input.
# missing key, wrong type, empty id or negative quantity -> ValueError. never return None.
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
    

    


# ---- try it ----
if __name__ == "__main__":
    # assert abs(safe_ratio(6, 3) - 2.0) < 1e-6 and abs(safe_ratio(0, 5) - 0.0) < 1e-6 and safe_ratio(1, 0) is None
    # assert normalize_event({"id": "e1", "quantity": 3}) == {'id': 'e1', 'quantity': 3}
    # normalize_event({"id": "e1", "quantity": -1})                 # raises ValueError
    # normalize_event({"quantity": 3})                              # raises ValueError
    pass
