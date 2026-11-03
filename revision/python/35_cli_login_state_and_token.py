# the godfather cli signs in through discord oauth in a browser, then shows a one-time token to paste.
# state: a random value saved in the browser session before redirect; the callback pops it and compares.
# popping makes the state single use. a missing, empty or different state means "not started here".
# issuing a new cli token revokes the member's older ones; the token is bound to one org and expires in 90 days.
# from platform: modules/godfather/api.py cli_sign_in(), cli_callback() (session["godfather_cli_login"]).
# from platform: modules/godfather/cli_login.py issue() revokes old kind=cli rows, member_for() checks kind, scope, org.
# note: this is authorization-code with a pasted token, not device-code polling. the cli never polls.
# explain: what attack does the state check stop? what would a stolen callback url do without it?
# explain: why is the token page sent with Cache-Control: no-store?

import secrets


def start_login(session: dict, org_prefix: str) -> str:
    # make a url-safe random state, store {"state", "org_prefix"} in session, return the state.
    pass


def finish_login(session: dict, state_param: str | None, code: str | None) -> str:
    # pop the stored login (missing -> {}). refuse with ValueError("not started here") unless it has a state,
    # state_param equals it (use secrets.compare_digest), and code is non empty. return the org_prefix.
    # a second finish with the same session must fail because the state was popped.
    pass


class TokenStore:
    def __init__(self):
        self.rows: list[dict] = []

    def issue(self, org_id: int, discord_id: str, now: float, days: int = 90) -> dict:
        # revoke (set revoked_at) every active row for the same org and member, then add a new row.
        pass

    def member_for(self, row: dict | None, org_id: int, now: float) -> str | None:
        # the discord id only if the row is active, unexpired, and belongs to org_id. else None.
        pass


if __name__ == '__main__':
    # assert: replay of a finished login fails, a wrong state fails, issuing twice leaves one active token,
    # a token for org 1 gives None when presented to org 2, an expired token gives None.
    pass
