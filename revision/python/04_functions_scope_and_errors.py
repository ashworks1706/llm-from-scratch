def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator

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
    

    
