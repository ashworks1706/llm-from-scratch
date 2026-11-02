# python starts with objects and names, not typed boxes.
# when x = 5, the name x refers to an integer object. assignment binds a name;
# it does not declare a fixed storage slot as it might in a lower-level language.
# start with integers, booleans, strings, None, comparisons and truthiness.
# then consider why 'if items:' checks non-emptiness rather than comparing to True.
# distinction: == compares values; 'is' checks object identity.
# in real code, use 'is None' to check the absence of a value.


def classify_price(price: float | None) -> str:
    # None -> 'missing', < 0 -> 'invalid', 0 -> 'free', otherwise 'paid'.
    # keep the branches exclusive and return one string on every path.
    pass


def clamp(value: float, low: float, high: float) -> float:
    # constrain a value to the closed interval [low, high].
    # think about whether low > high should raise an exception.
    pass
