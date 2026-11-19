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
# ex: "Alice.Smith" -> "alice.smith", "--..Bob!!" -> "bob", "!!!" -> "user", "a" * 30 -> "a" * 22
def safe_username(name: str) -> str:
    pass


# 2. accepts
# True only if pod_id is in cert.principals and valid_after <= now < valid_before
# (start inclusive, end exclusive); else False.
# ex: principals ("pod-a",), window [1000, 2000): "pod-a" at 1000, 1999 -> True;
#     at 999, 2000 -> False; "pod-b" -> always False
def accepts(cert: Cert, pod_id: str, now: int) -> bool:
    pass


if __name__ == '__main__':
    # check 1: a cert for pod a is rejected by pod b inside its window.
    # check 2: cert valid from now - BEFORE to now + AFTER is accepted at now + AFTER - 1 and
    # rejected at now + AFTER.
    pass
