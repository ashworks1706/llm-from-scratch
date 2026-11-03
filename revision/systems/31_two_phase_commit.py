# two-phase commit makes a distributed transaction atomic. phase 1: the coordinator sends PREPARE; each
# participant durably logs its vote (yes means it promises it CAN commit and holds its locks). phase 2:
# if every vote is yes the coordinator logs COMMIT and sends it, else logs ABORT. the coordinator's log
# write is the commit point. the known flaw: a participant that voted yes and lost the coordinator is
# BLOCKED; it cannot decide alone until the coordinator recovers.
# participant states: INIT -> PREPARED -> COMMITTED or ABORTED. coordinator: INIT -> WAITING -> COMMITTED/ABORTED.
# alternatives: saga (compensating actions, no locks, eventual) and idempotent retries via an outbox (see 28_).
# explain: why must a participant persist PREPARED before replying yes?
# explain: when would you choose a saga over 2pc for an order service that charges a card?

class Participant:
    def __init__(self, name: str, will_vote_yes: bool):
        # state starts at INIT; keep a durable-log list of state changes.
        pass

    def prepare(self) -> bool:
        # vote. a yes vote moves INIT -> PREPARED and logs it; a no vote moves straight to ABORTED.
        # prepare when already PREPARED returns True again (idempotent).
        pass

    def commit(self) -> None:
        # legal only from PREPARED (idempotent from COMMITTED). anything else raises.
        pass

    def abort(self) -> None:
        # legal from INIT, PREPARED, ABORTED. never from COMMITTED.
        pass


class Coordinator:
    def __init__(self, participants: list[Participant]):
        # log: list of decisions, written BEFORE phase 2 messages go out.
        pass

    def run(self, crash_after_decision: bool = False) -> str:
        # phase 1: collect votes (stop early on the first no, abort the rest). phase 2: log the decision,
        # then send it. if crash_after_decision, stop after logging and return "crashed".
        pass

    def recover(self) -> str:
        # read the log: decision found -> resend it to every participant; none found -> abort all.
        # return the final outcome. must be safe to call twice.
        pass


if __name__ == '__main__':
    # all yes -> everyone COMMITTED. one no -> everyone ABORTED. crash after logging commit leaves
    # participants PREPARED (blocked) until recover() commits them. assert no mixed outcomes ever.
    pass
