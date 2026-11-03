# a descriptor is an object in a class attribute with __get__, and optionally __set__ or __delete__.
# a data descriptor (has __set__) beats the instance __dict__; a non-data descriptor (only __get__) loses to it.
# property is a data descriptor; plain functions are non-data descriptors, which is how bound methods appear.
# __set_name__(owner, name) tells a descriptor which attribute it was assigned to, so it can store per-instance data.
# dunder methods plug objects into syntax: __len__ and __getitem__ give len() and indexing; __eq__ without __hash__ makes the class unhashable.
# from platform: modules/godfather/files.py PodFiles defines __enter__ and __exit__ so with PodFiles.open(...) closes the SFTP session.
# explain: why does a non-data descriptor that writes into the instance __dict__ run its __get__ only once per instance?
# explain: if you define __eq__ on a class, what happens to __hash__, and why is mutating a key after inserting it into a dict a bug?

from typing import Any


class Positive:
    # data descriptor: instance.attr = value must be a number > 0 or ValueError; reading returns the stored value.
    def __set_name__(self, owner: type, name: str) -> None:
        pass

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        # on the class itself (instance is None) return the descriptor.
        pass

    def __set__(self, instance: Any, value: float) -> None:
        pass


class Order:
    quantity = Positive()
    price = Positive()

    def __init__(self, quantity: float, price: float) -> None:
        # assign through the descriptors so validation runs on construction too.
        pass


class lazy:
    # non-data descriptor like cached_property: compute on first access, store in instance.__dict__ under the same name,
    # so later reads never reach __get__ again. the compute function is the decorated method.
    def __init__(self, func) -> None:
        pass

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        pass


class Ring:
    # fixed-capacity ring buffer: appending past capacity drops the oldest item.
    def __init__(self, capacity: int) -> None:
        pass

    def append(self, item: Any) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Any:
        # 0 is the oldest item; negative indexes count from the newest; out of range raises IndexError.
        pass

    def __iter__(self):
        # oldest to newest, a fresh iterator each call.
        pass

    def __repr__(self) -> str:
        pass
