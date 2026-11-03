# a response cache key must be identical for requests that mean the same thing and different for any that do not.
# query parameters are sorted so their order does not matter, but every repeated value for one name is kept.
# long keys are replaced by a stable digest so the backend never receives an oversized key.
# every key starts with a namespace so a whole family of entries can be dropped after new data is loaded.

import hashlib


def build_key(namespace: str, method: str, path: str, params: list[tuple[str, str]], max_len: int = 200) -> str:
    # params may repeat a name. return namespace:method:path:query with the pairs sorted.
    # if the result is longer than max_len, return namespace + ":hashed:" + the first 16 hex chars of its sha256.
    pass


class PrefixCache:
    def __init__(self):
        # store entries in a dict mapping key to (value, expires_at).
        pass

    def get(self, key: str, now: float):
        # return the value, or None when the key is missing or now >= expires_at, deleting expired entries.
        pass

    def set(self, key: str, value, ttl: float | None, now: float) -> None:
        # a ttl of None means the entry never expires.
        pass

    def invalidate_prefix(self, prefix: str) -> int:
        # remove every key starting with prefix and return how many were removed, O(n) in the number of keys.
        pass
