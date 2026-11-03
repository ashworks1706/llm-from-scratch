# rpc hides the network behind a function call; the cost is that it can fail in ways a call cannot
# (timeout, partial failure, retry duplicates). json-rpc 2.0: request {jsonrpc, method, params, id};
# no id = notification (no reply); errors use fixed codes (-32700 parse, -32600 invalid request,
# -32601 method not found, -32602 invalid params, -32603 internal); a JSON array is a batch.
# a path router maps method + path to a handler and extracts params like /orgs/:org/items/:id.
# from platform: modules/integrations/mcp_client.py sends json-rpc 2.0 requests (it builds
# {"jsonrpc": "2.0", "id": ..., "method": ..., "params": ...}); modules/mcp/server.py serves tools over MCP streamable HTTP.
# explain: why does an rpc call need an idempotency key if the client retries on timeout?

import json


class Dispatcher:
    def __init__(self):
        # method name -> callable. params may be a list (positional) or dict (named).
        pass

    def register(self, name: str, fn) -> None:
        pass

    def handle(self, raw: str) -> str | None:
        # parse one request or a batch. return the JSON response text, or None for a pure notification
        # (or an all-notification batch). a handler exception -> -32603 with the SAME id, never a crash.
        # empty batch -> single -32600. one bad item in a batch must not fail the others.
        pass


class Router:
    def __init__(self):
        pass

    def add(self, method: str, pattern: str, handler) -> None:
        # pattern like "/orgs/:org/items/:id". a static segment beats a param at the same position.
        pass

    def match(self, method: str, path: str):
        # return (handler, params dict) or None. trailing slash ignored. 405 vs 404: return the set of
        # allowed methods when the path matches but the method does not (design the return type).
        pass


if __name__ == '__main__':
    # assert a batch of [valid call, notification, unknown method] returns exactly two responses.
    # assert "/orgs/soda/items/7" extracts {"org": "soda", "id": "7"} and "/orgs/soda/items/new"
    # prefers a static "new" route over :id.
    pass
