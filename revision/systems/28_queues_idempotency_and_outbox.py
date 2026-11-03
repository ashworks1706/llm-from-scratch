# a queue gives at-least-once delivery: a consumer must ack AFTER finishing, so a crash redelivers the
# message. exactly-once processing is built from at-least-once delivery + an idempotent consumer
# (dedupe by idempotency key); exactly-once delivery over a network is impossible.
# the dual-write problem: updating the db and publishing a message are two systems, so one can fail.
# the outbox pattern writes the message to an outbox table in the SAME db transaction as the state
# change; a relay polls the outbox, publishes, and marks it sent (so the publish itself is at-least-once).
# a lease/visibility timeout lets a crashed consumer's message become available again.
# stdlib only: sqlite3 or plain dicts.
# from halo: packages/db/src/schema.ts defines singular_delivery (id = "rc:" + RevenueCat event id, status
# pending/sending/delivered/skipped/failed/ambiguous/shadow, lease_token, lease_expires_at, next_attempt_at,
# send_attempts). packages/api/src/singular/SingularPurchaseForwarder.ts inserts the row if absent, claims
# it under a lease (LEASE_MS = 10 min), retries retryable failures with backoff up to MAX_SEND_ATTEMPTS = 8,
# and parks a failure with mayHaveApplied (and expired leases) as ambiguous. plans/010 says Singular does
# not dedupe, so a blind resend is permanent data corruption. mirror claim, lease and ambiguous on a toy.
# explain: why is "ambiguous" safer than retrying when a send times out after the request left?

import time


class Queue:
    def __init__(self, visibility_timeout: float, now=time.monotonic):
        pass

    def send(self, msg_id: str, body: str) -> None:
        pass

    def receive(self) -> tuple[str, str, str] | None:
        # return (msg_id, body, lease_token) for the oldest visible message and hide it until
        # now + visibility_timeout. None if nothing is visible.
        pass

    def ack(self, msg_id: str, lease_token: str) -> bool:
        # delete only with the CURRENT lease token (a stale holder cannot ack a redelivered message).
        pass


class IdempotentConsumer:
    def __init__(self):
        # processed ids seen so far; effects must happen once per id even on redelivery.
        pass

    def handle(self, msg_id: str, body: str, effect) -> bool:
        # run effect(body) once per msg_id; return True if it ran, False if it was a duplicate.
        # think about the crash window between effect and recording the id; put both in one txn in real life.
        pass


class Outbox:
    def __init__(self, conn):
        # sqlite3 connection. tables: accounts(id, balance) and outbox(id, payload, sent).
        pass

    def transfer_and_enqueue(self, account_id: int, delta: int, payload: str) -> None:
        # update the balance AND insert the outbox row in ONE transaction; roll back both on failure.
        pass

    def relay(self, publish) -> int:
        # publish every unsent row in id order, mark sent after publish succeeds. return how many.
        pass
