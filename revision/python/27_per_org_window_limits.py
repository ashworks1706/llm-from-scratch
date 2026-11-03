# two kinds of per-org limit. a count limit caps what the org keeps (pods): usage is counted live.
# a rate limit caps use per fixed window (calls a minute): one counter row per (org, key, window_start).
# effective value order: org override, then env, then declared default. None means unlimited.
# consume() adds to the counter only if the add stays within the limit; an over-limit call counts nothing.
# from platform: core/limits.py window_start(), value(), check() (409), consume() (429 + upsert counter).
# from platform: modules/mcp/runtime.py CALLS_LIMIT mcp.calls_per_minute is consumed on every tool call.
# explain: used() then upsert is two steps. where can two concurrent callers exceed the limit?
# explain: fixed windows allow 2x the limit across a boundary. what would a token bucket change?

import sqlite3
from datetime import datetime, timedelta


def window_start(window: str, now: datetime) -> datetime:
    # truncate now to the minute, hour or day. unknown window name -> ValueError.
    pass


def effective_limit(override: dict[str, int | None], env: dict[str, str], default: int | None, key: str) -> int | None:
    # override wins even when its value is None (explicit unlimited). then env[key]: "unlimited" -> None,
    # a number -> max(0, n), garbage -> default. then default.
    pass


def make_counters() -> sqlite3.Connection:
    # usage_counters(org_id, key, window_start text, count int) with UNIQUE(org_id, key, window_start).
    pass


def consume(db: sqlite3.Connection, org_id: int, key: str, window: str, limit: int | None, amount: int, now: datetime) -> None:
    # amount <= 0 is a no-op. over the limit -> raise OverLimit and write nothing.
    # else INSERT ... ON CONFLICT DO UPDATE SET count = count + amount.
    pass


class OverLimit(Exception):
    pass


if __name__ == '__main__':
    # assert: 3 calls at limit 3 pass and the 4th raises, a new minute resets,
    # limit None never raises, org 2 has its own counter, and a rejected amount leaves count unchanged.
    pass
