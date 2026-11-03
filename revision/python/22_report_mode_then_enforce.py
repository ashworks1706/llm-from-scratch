# rolling out a new authorization check on a live system: log it first, enforce it later.
# in report mode a failing check logs "would_deny" and lets the request through.
# flip one flag and the same decision becomes a 403. run report mode until the log is clean.
# some checks never get report mode (superadmin routes change orgs and limits).
# from platform: modules/auth/access.py decide(), org_officer_denial(), superadmin_denial(), enforcing().
# from platform: core/config.py ACCESS_ENFORCE env flag, default false.
# explain: what does a clean report-mode log prove, and what can it not prove?
# explain: why does bot_unavailable (cannot verify) deny with 503 and not 403 once enforcing?

from dataclasses import dataclass, field


@dataclass
class Decider:
    enforce: bool
    log: list[str] = field(default_factory=list)

    def decide(self, reason: str) -> bool:
        # append "deny:<reason>" when enforcing else "would_deny:<reason>".
        # return True only when the request must be refused.
        pass


def officer_denial(
    d: Decider, discord_id: str | None, org_guild: str, officer_guilds: set[str] | None, superadmins: set[str]
) -> tuple[str, int] | None:
    # order: no discord id -> superadmin passes -> guilds None (cannot verify) -> not an officer of org_guild.
    # return (message, status) only when d.decide(...) says refuse; else None.
    pass


def superadmin_denial(discord_id: str | None, superadmins: set[str]) -> tuple[str, int] | None:
    # always refuses non superadmins, whatever the enforce flag is.
    pass


if __name__ == '__main__':
    # assert report mode returns None but logs would_deny, enforce mode returns 403,
    # unknown guild set (None) gives 503 only when enforcing, superadmin_denial ignores the flag.
    pass
