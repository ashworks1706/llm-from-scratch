# a dataclass generates __init__, __repr__ and __eq__ from annotated fields; frozen=True forbids assignment and adds __hash__.
# a mutable default like items: list = [] is rejected; use field(default_factory=list) so each instance gets its own list.
# __post_init__ is the place for invariants. __slots__ (or slots=True) removes the per-instance __dict__: less memory, no new attributes.
# a frozen dataclass is a value object: equal fields mean equal and hashable, so it can key a dict or sit in a set.
# dataclasses.replace(obj, field=value) is how you "change" a frozen instance: it builds a new one and reruns __post_init__.
# from platform: core/limits.py declares Limit as @dataclass(frozen=True) and core/webhooks.py does the same for its records.
# explain: why can a frozen dataclass still hold a list that changes, and how would you stop that?
# explain: what breaks when a class with __slots__ inherits from a class without __slots__?

import sys
from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class Window:
    start: int
    end: int

    def __post_init__(self) -> None:
        # reject end < start with ValueError. an empty window (start == end) is allowed.
        # with frozen=True you cannot assign here; you only validate.
        pass

    def overlaps(self, other: "Window") -> bool:
        # half-open [start, end): windows that only touch at a boundary do not overlap. empty windows overlap nothing.
        pass

    def merged(self, other: "Window") -> "Window":
        # the smallest window covering both; raise ValueError if they neither overlap nor touch.
        pass


@dataclass
class Bag:
    # a name, a list of ints and a set of tags; the list and the set default to empty and are never shared between instances.
    # first write items: list[int] = [] and read the error, then fix it.
    pass


def shifted(window: Window, by: int) -> Window:
    # a new window moved by `by`, using dataclasses.replace. the original is unchanged.
    pass


def instance_bytes(obj: object) -> int:
    # sys.getsizeof(obj) plus sys.getsizeof(obj.__dict__) when the instance has a __dict__. compare Window to a plain class.
    pass
