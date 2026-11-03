# dns resolution: stub resolver -> recursive resolver (cache) -> root -> TLD -> authoritative server.
# answers carry a TTL; caches must honour it, and negative answers (NXDOMAIN) are cached too.
# tls 1.3 in short: client hello (supported groups, key share) -> server hello + key share ->
# server certificate chain + signature -> both derive keys -> finished. the client checks the chain
# to a trusted root, that the hostname matches the SAN, and the validity window.
# the toy models the DECISIONS (cache with ttl, validity and hostname checks), not the crypto.
# from platform: core/net.py resolves a host and connects only to an address that passed the public-ip
# check, so a second dns answer cannot swap in a private address (dns rebinding / ssrf).
# explain: why does connecting by the checked IP matter more than checking then re-resolving?

import ipaddress
from dataclasses import dataclass


@dataclass
class Cert:
    subject_alt_names: list[str]
    not_before: int
    not_after: int
    issuer: str


class TtlResolver:
    def __init__(self, upstream, clock):
        # upstream(name) -> (list[str] addresses, ttl_seconds) or None for NXDOMAIN; clock() -> now.
        pass

    def resolve(self, name: str) -> list[str] | None:
        # cache hits until expiry; also cache NXDOMAIN for a short fixed time. lowercase the name.
        pass


def hostname_matches(host: str, pattern: str) -> bool:
    # exact match, or a leading "*." wildcard covering exactly ONE label ("a.b.example.com" is not
    # matched by "*.example.com"). case-insensitive.
    pass


def cert_ok(cert: Cert, host: str, now: int, trusted_issuers: set[str]) -> str | None:
    # return None if fine, else a reason: "expired", "not_yet_valid", "hostname", "untrusted".
    pass


def is_public_address(ip: str) -> bool:
    # false for private, loopback, link-local, multicast, reserved; use the ipaddress module.
    pass
