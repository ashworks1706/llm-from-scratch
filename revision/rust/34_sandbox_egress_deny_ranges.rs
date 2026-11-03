// the sandbox reaches the internet only through a proxy. the proxy is a DENY list, not an allowlist:
// ports 80 and 443 only, CONNECT only to 443, and any destination in a private, loopback, link-local,
// carrier-nat, multicast or reserved range is refused. everything else public is allowed.
// the purpose is to stop a model-written script from reaching internal services (ssrf), not to curate sites.
//
// from sparky: deploy/sandbox/squid.conf (acl internal dst ..., http_access deny rules) and
// apps/engine/src/runtime/tools/sandbox/egress.rs (internal docker network + proxy container).
// toy: ipv4 only, a subset of the squid ranges, as (network, prefix_len).
//
// explain: why is a deny list of ranges acceptable here when it is a weak design for site filtering?
// explain: a hostname resolves to 127.0.0.1 after the check. which layer must check the resolved address?

const DENIED: [([u8; 4], u8); 11] = [
    ([0, 0, 0, 0], 8), ([10, 0, 0, 0], 8), ([100, 64, 0, 0], 10), ([127, 0, 0, 0], 8),
    ([169, 254, 0, 0], 16), ([172, 16, 0, 0], 12), ([192, 168, 0, 0], 16),
    ([198, 18, 0, 0], 15), ([224, 0, 0, 0], 4), ([240, 0, 0, 0], 4), ([192, 0, 0, 0], 24),
];

fn in_range(ip: [u8; 4], network: [u8; 4], prefix: u8) -> bool {
    // compare the first prefix bits. prefix 0 matches everything, 32 matches one host.
    // do it with u32 masks, not string matching.
    todo!()
}

fn is_denied_ip(ip: [u8; 4]) -> bool {
    // true when the ip falls in any DENIED range.
    todo!()
}

fn allow_request(is_connect: bool, port: u16, ip: [u8; 4]) -> bool {
    // deny: port not in {80, 443}; CONNECT to a port other than 443; denied ip. otherwise allow.
    todo!()
}

fn main() {
    // tests: 172.15.255.255 allowed but 172.16.0.0 and 172.31.255.255 denied; 100.64.0.1 denied; 8.8.8.8 allowed.
}
