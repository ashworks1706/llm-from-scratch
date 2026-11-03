# an ssh user certificate is a signed statement: this key, these principals, valid from A to B.
# the pod trusts the org's CA key, so there is no per-user key list on the pod to clean up.
# revocation is expiry: platform issues short lived certs (valid_after = now - 5 min for clock skew,
# valid_before = now + 12 hours) and keeps no revocation list. a removed member simply cannot get a new one.
# members get a forced command that locks them into their own account; officers get none.
# from platform: modules/godfather/ssh.py sign_user_key(), pod_principal() = gf-<pod id>, safe_username().
# from platform: CERT_BEFORE, CERT_AFTER, LOGIN_COMMAND constants in the same file. no cert is stored, only a connections row.
# explain: why back-date valid_after by 5 minutes? what fails without it?
# explain: what is the worst case between "officer removes access" and "access really ends"?

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
    # lowercase, keep only [a-z0-9._-], strip leading . and -, cut to 22 chars, "user" if empty.
    pass


def issue(pod_id: str, discord_id: str, username: str, is_admin: bool, now: int) -> Cert:
    # principal is "gf-" + pod_id (exactly one). key_id "discord:<id>:<safe username>".
    # window [now - BEFORE, now + AFTER]. force_command only for members: "<login cmd> <safe username>".
    pass


def accepts(cert: Cert, pod_id: str, now: int) -> bool:
    # what the pod checks: principal matches this pod and valid_after <= now < valid_before.
    # model the boundaries explicitly.
    pass


if __name__ == '__main__':
    # assert: a cert for pod a is rejected by pod b, accepted at now and at now+AFTER-1 and rejected at now+AFTER,
    # accepted 4 minutes in the past (skew) but not 6, admin has no force_command, member has one,
    # safe_username("../Evil User!") has no slash or space.
    pass
