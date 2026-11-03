# tcp is a byte stream, not a message stream: one send() may arrive as many recv()s, and many sends may
# merge into one. you need framing. length-prefixed framing: a fixed-size big-endian length, then that
# many bytes. delimiter framing needs escaping. always loop until you have exactly n bytes.
# also cap the length so a hostile 4 GB prefix cannot allocate your memory.
# stdlib only: socket, struct, threading.
# explain: what does Nagle's algorithm do and when does it add 40 ms of latency?
# explain: how do you detect a half-open connection that the peer never closed?

import socket
import struct

MAX_FRAME = 1 << 20


def recv_exact(sock: socket.socket, n: int) -> bytes:
    # loop on recv until n bytes arrive. EOF before n -> raise ConnectionError. n == 0 -> b"".
    pass


def send_frame(sock: socket.socket, payload: bytes) -> None:
    # struct.pack(">I", len) + payload with sendall. refuse payloads over MAX_FRAME.
    pass


def recv_frame(sock: socket.socket) -> bytes | None:
    # return None on clean EOF at a frame boundary. raise on EOF mid-frame or length > MAX_FRAME.
    pass


def echo_server(listener: socket.socket, max_clients: int) -> None:
    # accept up to max_clients, one thread each, echo frames until the client closes.
    pass


if __name__ == '__main__':
    # use socket.socketpair() for unit checks. send three frames in one sendall to prove they are
    # split correctly, and one frame dribbled one byte at a time to prove recv_exact loops.
    pass
