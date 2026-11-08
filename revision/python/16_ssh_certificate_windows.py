# An ssh user certificate is a signed statement that a key holds certain principals from time a to
# time b. The server trusts the signing authority, so there is no per user key list to clean up.
# Revocation is expiry, so certificates are short lived and a removed member cannot get a new one.
# valid_after is set a few minutes in the past so that a client or server with a slow clock still
# accepts a fresh certificate.

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


# Problem 1: make a safe username.
#
# The function takes name, any string, and returns a username that is safe to use on a server. It
# lowercases the text, keeps only the characters a to z, 0 to 9, dot, underscore and hyphen, removes
# any dots and hyphens at the start, and cuts the result to at most 22 characters. If nothing is
# left, it returns "user". Apply the steps in that order.
#
# Example: "Alice.Smith" gives "alice.smith", "--..Bob!!" gives "bob", "!!!" gives "user", and a
# string of 30 letters "a" gives 22 letters "a".
def safe_username(name: str) -> str:
    pass


# Problem 2: does a pod accept this certificate.
#
# The function takes cert, a Cert, pod_id, a string, and now, the current time as an int number of
# seconds. It returns True only if pod_id is one of the certificate's principals and the time is
# inside the validity window, meaning valid_after <= now < valid_before. The start is inclusive and
# the end is exclusive. Otherwise it returns False.
#
# Example: for a certificate with principals ("pod-a",), valid_after 1000 and valid_before 2000,
# accepts gives True for pod "pod-a" at times 1000 and 1999, and False at time 999, at time 2000, and
# for pod "pod-b" at any time.
def accepts(cert: Cert, pod_id: str, now: int) -> bool:
    pass


if __name__ == '__main__':
    # Check one: a certificate issued for pod a is rejected by pod b at a time inside its window.
    # Check two: build a certificate valid from now - BEFORE to now + AFTER, then assert it is
    # accepted at now + AFTER - 1 and rejected at now + AFTER.
    pass
