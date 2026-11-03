# godfather authorization is layered. who are you (officer, member, cli token), are you in the org's
# discord server, may you reach THIS pod, is the pod up, and only then is a certificate signed.
# a pod is reachable if it is public or the member's discord id is in allowed_users. officers bypass.
# the checks run in a fixed order and each failure has its own status, so a client can tell why.
# from platform: modules/godfather/service.py connect(), _may_connect(), accessible_pods(), _users().
# from platform: modules/godfather/api.py _member_route (cli token -> member_for -> _is_member) and connect route.
# explain: why 404 for a pod of another org but 403 for a pod of your own org you may not use?
# explain: why re-check discord membership on every cli request instead of trusting the 90 day token?

from dataclasses import dataclass, field


class GodfatherError(Exception):
    def __init__(self, message: str, status: int = 400):
        pass


@dataclass
class Pod:
    pod_id: str
    org_id: int
    is_public: bool = False
    allowed_users: list[str] = field(default_factory=list)
    status: str = "RUNNING"
    ssh_address: tuple[str, int] | None = ("203.0.113.5", 40022)


def clean_allowed(value: object, limit: int = 500) -> list[str]:
    # must be a list of at most limit items; each str(item) is all digits and 5..25 long.
    # return a sorted de-duplicated list. anything else -> GodfatherError 400.
    pass


def may_connect(pod: Pod, discord_id: str) -> bool:
    # public, or discord_id in allowed_users.
    pass


def connect(pods: list[Pod], org_id: int, pod_id: str, discord_id: str, is_admin: bool, key_ok: bool) -> tuple[str, int]:
    # order: key_ok else 400 -> pod found in THIS org else 404 -> not admin and not may_connect else 403
    # -> status != RUNNING else 409 -> ssh_address None else 503. return the ssh address when all pass.
    pass


if __name__ == '__main__':
    # assert each status code above with a table of cases, including a pod of org 2 requested from org 1 (404),
    # an admin on a private pod (allowed), and clean_allowed(["12345","12345"]) == ["12345"].
    pass
