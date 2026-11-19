# same meaning -> same key, different meaning -> different key. sort params but keep repeats; hash
# long keys; prefix every key with a namespace so a family can be dropped after a data load.

import hashlib


# 1. build_key
# params: (name, value) pairs, names may repeat. key = namespace:method:path:query where query is
# pairs sorted by name then value, as name=value joined by "&". no params -> key ends with ":".
# len(key) > max_len -> namespace + ":hashed:" + first 16 hex chars of sha256(full key utf-8).
# len == max_len is kept as is. input list unchanged.
# ex: ("v1", "GET", "/items", [("b", "2"), ("a", "1"), ("a", "0")]) -> "v1:GET:/items:a=0&a=1&b=2",
#     any pair order -> same key
def build_key(namespace: str, method: str, path: str, params: list[tuple[str, str]], max_len: int = 200) -> str:
    pass


# 2. PrefixCache
# string keys; caller passes now (seconds), never read a clock. entry expires when now >= expires_at.
# ex: set("k", 1, 10, now=0) -> get("k", now=9) -> 1, get("k", now=10) -> None
class PrefixCache:
    # store: dict key -> (value, expires_at).
    def __init__(self):
        pass

    # value for key, or None if missing or now >= expires_at. delete an expired entry when found.
    def get(self, key: str, now: float):
        pass

    # store value, replacing any old entry; expires at now + ttl. ttl None -> never expires.
    def set(self, key: str, value, ttl: float | None, now: float) -> None:
        pass

    # remove every key starting with prefix, return the count. O(n) scan. "" removes everything.
    def invalidate_prefix(self, prefix: str) -> int:
        pass
