# cached org reads carry an ETag (hash of the body). the client revalidates every time and gets 304
# with no body when the tag still matches: the server saves bytes, the browser keeps freshness.
# after any successful write (POST PUT PATCH DELETE under /api/, status < 400) every cached "org" read is dropped.
# the access check runs BEFORE the cache is read, so the cache never serves a caller the route would refuse.
# from platform: core/http/cached.py _encode(), conditional_json(), cached_json(), register_cache_invalidation().
# from platform: org_key(prefix, *parts) = ("org", prefix, ...) so one prefix drops one org or all orgs.
# explain: why is "Vary: Authorization, Cookie" needed on a cached private response?
# explain: why drop all orgs' reads on one org's write instead of just that org's? what does it cost?

import hashlib
import json

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def etag_of(payload: object) -> str:
    # sha256 hex of the canonical JSON bytes (sorted keys, no spaces), first 32 chars.
    # equal payloads must give equal tags regardless of key order.
    pass


def conditional(payload: object, if_none_match: str | None) -> tuple[int, dict, bytes]:
    # 304, headers with ETag, empty body when if_none_match equals the tag (ignore surrounding quotes).
    # else 200 with the body, the ETag, and Cache-Control "private, no-cache".
    pass


def should_invalidate(method: str, status: int, path: str) -> bool:
    # True only for a write method, status below 400, and a path starting with /api/.
    pass


def org_key(prefix: str, *parts) -> tuple:
    # ("org", prefix, *parts)
    pass


if __name__ == '__main__':
    # assert: key order does not change the tag, a one-field change does, matching tag gives 304 and empty body,
    # a failed write (500) or a GET never invalidates, a write to /health does not.
    pass
