# http/1.1 is text: request line, headers, blank line (CRLF CRLF), then a body whose length comes from
# content-length or chunked encoding. connections are keep-alive by default; the server must find
# the end of one request precisely to start parsing the next on the same socket (pipelining).
# websocket upgrade: the server answers 101 with Sec-WebSocket-Accept =
# base64(sha1(client_key + fixed GUID 258EAFA5-E914-47DA-95CA-C5AB0DC85B11)).
# stdlib only: base64, hashlib.
# explain: what is request smuggling and why do content-length plus transfer-encoding conflicts cause it?
# explain: what does head-of-line blocking look like in http/1.1 and how does http/2 change it?

from dataclasses import dataclass, field


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


def parse_request(buf: bytes) -> tuple[Request, int] | None:
    # return (request, bytes_consumed) or None if the buffer holds no complete request yet.
    # header names are case-insensitive: store lowercase. body = exactly content-length bytes.
    # reject malformed request lines and duplicate content-length with ValueError.
    pass


def wants_keep_alive(req: Request) -> bool:
    # http/1.1 default true unless "connection: close"; http/1.0 default false unless keep-alive.
    pass


def parse_pipeline(buf: bytes) -> list[Request]:
    # parse every complete request in buf back to back; ignore a trailing partial one.
    pass


def websocket_accept(client_key: str) -> str:
    # compute the Sec-WebSocket-Accept value.
    pass


if __name__ == '__main__':
    # assert the RFC 6455 example: key dGhlIHNhbXBsZSBub25jZQ== -> s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
    # also feed a request split across two buffers and assert None first, then a full parse.
    pass
