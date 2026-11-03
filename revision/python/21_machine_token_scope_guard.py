# a machine token is an opaque secret shown once. the db keeps only its sha256 hash.
# verify = hash the presented value, look it up, then reject revoked and expired rows.
# order of refusals matters: no/invalid token -> 401, valid token without the scope -> 403,
# valid token for a different org than the url -> 403.
# from platform: modules/auth/machine_tokens.py issue(), verify(), rotate().
# from platform: modules/auth/decorators.py machine_scope_required() wraps a route in these three checks.
# explain: why store a hash and not the token? why does a leaked db row not leak access?
# explain: why does verify throttle the last_used_at write to once a minute?

import hashlib
import secrets
from dataclasses import dataclass

PREFIX = "plat_"


@dataclass
class TokenRow:
    token_hash: str
    org_id: int
    scopes: frozenset[str]
    expires_at: float | None = None
    revoked_at: float | None = None


def issue(store: dict[str, TokenRow], org_id: int, scopes: list[str], now: float, ttl: float | None) -> str:
    # make PREFIX + random value, store only its sha256 hex in store keyed by hash.
    # return the plain value once. expires_at is now + ttl, or None for no expiry.
    pass


def verify(store: dict[str, TokenRow], presented: str | None, now: float) -> TokenRow | None:
    # None for missing value, wrong prefix, unknown hash, revoked, or expires_at <= now.
    # the expiry boundary is exclusive: a token expiring exactly now is dead.
    pass


def guard(store: dict[str, TokenRow], presented: str | None, scope: str, url_org: int, now: float) -> tuple[int, str]:
    # return (200, "ok") or (401, ...) / (403, ...) following the order in the header.
    pass


if __name__ == '__main__':
    # assert: revoked and expired tokens give 401, wrong scope gives 403,
    # a token from org 1 on a url for org 2 gives 403, and the store never holds the plain value.
    pass
