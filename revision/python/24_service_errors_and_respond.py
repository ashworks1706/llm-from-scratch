# a service takes a db session and plain values and raises a typed error that carries its own status.
# the edge (route, tool, bot) never sees flask or http inside the service. it only translates the error.
# on a ServiceError the edge rolls the session back, then answers {"error": msg} with e.status.
# an upstream provider error is re-typed: provider 4xx -> 400 (your request), anything else -> 424.
# from platform: core/errors.py ServiceError(message, status=400).
# from platform: modules/auth/routes.py respond(db, view, ...) and modules/godfather/service.py _error(), _call().
# explain: why must respond() roll back before returning, and who commits on success?
# explain: why 424 for a provider outage and not 500 or 502?


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 400):
        pass


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        pass


class FakeSession:
    def __init__(self):
        self.rolled_back = 0
        self.committed = 0

    def rollback(self):
        self.rolled_back += 1

    def commit(self):
        self.committed += 1


def retype(e: ProviderError) -> ServiceError:
    # status None or outside 400..499 -> 424. a 4xx from the provider -> 400. keep the message.
    pass


def respond(db: FakeSession, view, *args, **kwargs) -> tuple[dict, int]:
    # call view(db, *args, **kwargs). return (result, 200) on success without committing here.
    # on ServiceError: db.rollback() once and return ({"error": message}, status).
    # any other exception must propagate untouched.
    pass


if __name__ == '__main__':
    # assert: provider 404 -> 400, provider 503 -> 424, provider None -> 424,
    # a failing view rolls back exactly once and a succeeding one never rolls back,
    # a KeyError from the view is not swallowed.
    pass
