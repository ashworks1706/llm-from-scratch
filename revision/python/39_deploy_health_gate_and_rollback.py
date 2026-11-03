# a deploy is done only when the new containers are healthy AND /health reports the new commit.
# waiting is a small state machine: healthy or running -> ok, unhealthy exited dead -> fail now,
# starting or unknown -> keep polling until a timeout. a hung container must not block the deploy forever.
# bookkeeping files make rollback possible: deployed-sha is the current commit, rollback-sha the previous one.
# rollback-sha is written only when the prior sha is a valid 40 hex sha and differs from the new one.
# from platform: Makefile deploy target wait_for_health(), the /health commit check, .deploy/deployed-sha, .deploy/rollback-sha.
# from platform: Makefile rollback target restores the baseline sha; core/host.py agent_snapshot() marks a stale agent file.
# explain: why check the commit in /health and not just that the container is up?
# explain: why write state files to a tmp name and rename them?

import re

SHA = re.compile(r"^[0-9a-f]{40}$")


def wait_for_health(poll, timeout: float, step: float = 2.0) -> bool:
    # poll() returns a status string each call. "healthy"/"running" -> True, "unhealthy"/"exited"/"dead" -> False,
    # anything else sleeps conceptually (advance elapsed by step) and retries. elapsed >= timeout -> False.
    # no real sleeping: count elapsed instead.
    pass


def health_reports_commit(body: str, new_sha: str) -> bool:
    # True if new_sha is in the health body and new_sha is a valid 40 hex sha.
    pass


def record_deploy(files: dict[str, str], new_sha: str) -> None:
    # files maps file name -> contents. refuse (ValueError) a non sha. if deployed-sha holds a valid prior sha
    # different from new_sha, store it as rollback-sha. then set deployed-sha = new_sha.
    pass


def rollback(files: dict[str, str]) -> str:
    # restore deployed-sha from rollback-sha, delete rollback-sha, return the restored sha.
    # no valid rollback-sha -> RuntimeError("no previous version").
    pass


def is_stale(generated_at: int, now: int, stale_after: int = 120) -> bool:
    # a host agent snapshot older than stale_after seconds means the agent stopped. boundary is exclusive.
    pass


if __name__ == '__main__':
    # assert: starting, starting, healthy -> True; starting forever -> False at the timeout;
    # exited on the first poll -> False; redeploying the same sha keeps the old rollback-sha;
    # two deploys then rollback returns the first sha.
    pass
