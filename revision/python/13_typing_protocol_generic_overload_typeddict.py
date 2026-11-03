# a Protocol is structural: any object with the right methods matches, no inheritance needed (duck typing a checker can verify).
# Generic classes and functions keep the relation between input and output types (def first[T](xs: list[T]) -> T | None).
# @overload gives a checker several signatures for one runtime function; only the last, undecorated def has a body.
# Literal narrows a string parameter to specific values. isinstance, None checks and TypeGuard narrow a union inside a branch.
# TypedDict describes dict keys for a checker only; at runtime it is a plain dict, so untrusted data still needs validating.
# from platform: core/hosting.py defines HostingClient and HostingProvider as Protocols; providers register into PROVIDERS by name.
# from platform: make check and make ci both run uv run ty check . ; types guide the checker and never validate JSON.
# explain: what does a Protocol buy you over an ABC when the implementations live in different packages?
# explain: why can a TypedDict-annotated dict from json.loads still contain a string where the type says int?

from collections.abc import Iterable
from typing import Literal, NotRequired, Protocol, TypedDict, overload


class SupportsClose(Protocol):
    def close(self) -> None: ...


def close_all(resources: Iterable[SupportsClose]) -> list[Exception]:
    # call close() on every resource even when one raises; return the exceptions in order. never stop at the first.
    pass


class Stack[T]:
    def push(self, item: T) -> None:
        # O(1).
        pass

    def pop(self) -> T:
        # raise IndexError("pop from empty stack") when empty, else the newest item.
        pass

    def peek(self) -> T | None:
        # newest item without removing it, None when empty.
        pass


@overload
def convert(value: str, kind: Literal["int"]) -> int: ...
@overload
def convert(value: str, kind: Literal["float"]) -> float: ...
def convert(value: str, kind: str) -> int | float:
    # int(value) or float(value) by kind; any other kind raises ValueError. keep the overloads in sync with the body.
    pass


def total_length(value: str | list[str] | None) -> int:
    # None -> 0, str -> its length, list -> sum of lengths. narrow with isinstance, no casts.
    pass


class OrderRow(TypedDict):
    id: str
    qty: int
    note: NotRequired[str]


def parse_row(raw: dict[str, object]) -> OrderRow:
    # check id is a non-empty str and qty is an int (not bool) >= 0; note, if present, must be a str. raise ValueError otherwise.
    pass
