# an ssh user cert says a key holds some principals from time a to b. revocation is expiry, so certs
# are short lived. valid_after is set a few minutes back to tolerate slow clocks.

import re
from dataclasses import dataclass

BEFORE = 5 * 60
AFTER = 12 * 3600


@dataclass(frozen=True)
class Cert:
    key_id: str
    principals: tuple[str, ...]
    valid_after: int
    valid_before: int
    force_command: str | None


# 1. safe_username
# in this order: lowercase, keep only [a-z0-9._-], strip leading dots and hyphens, cut to 22 chars.
# nothing left -> "user".
def safe_username(name: str) -> str:
    pass


# 2. accepts
# True only if pod_id is in cert.principals and valid_after <= now < valid_before
# (start inclusive, end exclusive); else False.
def accepts(cert: Cert, pod_id: str, now: int) -> bool:
    pass


# ---- try it ----
if __name__ == '__main__':
    # print(safe_username("Alice.Smith"))           # alice.smith
    # print(safe_username("--..Bob!!"))             # bob
    # print(safe_username("!!!"))                   # user
    # print(safe_username("a" * 30))                # "a" * 22
    # cert = Cert("k1", ("pod-a",), 1000, 2000, None)
    # print(accepts(cert, "pod-a", 999))            # False
    # print(accepts(cert, "pod-a", 1000))           # True
    # print(accepts(cert, "pod-a", 1999))           # True
    # print(accepts(cert, "pod-a", 2000))           # False
    # print(accepts(cert, "pod-b", 1500))           # False
    pass
