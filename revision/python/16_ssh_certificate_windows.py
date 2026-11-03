# an ssh user certificate is a signed statement that a key holds these principals from time a to time b.
# the server trusts the signing authority, so there is no per-user key list to clean up.
# revocation is expiry, so certificates are short lived and a removed member cannot get a new one.
# valid_after is back-dated a few minutes so a client or server with a slow clock still accepts a fresh certificate.

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


def safe_username(name: str) -> str:
    # lowercase, keep only [a-z0-9._-], strip leading dots and dashes, and cut to 22 chars.
    # return "user" if nothing is left.
    pass


def accepts(cert: Cert, pod_id: str, now: int) -> bool:
    # the principal must match this pod and valid_after <= now < valid_before.
    # the upper bound is exclusive.
    pass


if __name__ == '__main__':
    # assert that a certificate for pod a is rejected by pod b.
    # assert it is accepted at now + AFTER - 1 and rejected at now + AFTER.
    pass
