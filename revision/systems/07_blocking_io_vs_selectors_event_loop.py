# blocking I/O parks a thread per connection: simple, but threads cost memory and context switches.
# an event loop uses one thread plus the OS readiness API (epoll/kqueue, wrapped by selectors) to ask
# "which sockets are ready", then does only non-blocking reads/writes. never block inside the loop.
# readiness (can read now) is not the same as completion (io_uring style); reads may return less than asked.
# stdlib only: selectors, socket.
# explain: why does one slow handler stall every client in an event loop but not in thread-per-connection?
# explain: what is the c10k problem and what breaks first at 10k threads?

import selectors
import socket


def serve_echo(host: str, port: int, stop_after_connections: int) -> None:
    # selectors.DefaultSelector, listening socket non-blocking registered for EVENT_READ.
    # on accept: set non-blocking, register the client. on readable: recv; empty bytes = peer closed,
    # so unregister and close. echo with send; handle a partial send by buffering the rest and
    # switching interest to EVENT_WRITE until drained. return after N connections closed.
    pass


def blocking_echo_threaded(host: str, port: int, stop_after_connections: int) -> None:
    # same contract using one thread per accepted socket, blocking recv/sendall.
    pass


def echo_once(host: str, port: int, payload: bytes) -> bytes:
    # client helper: connect, send payload, shutdown(SHUT_WR), read until EOF, return bytes.
    pass


if __name__ == '__main__':
    # bind a socket to port 0 to get a free port, start serve_echo in a thread, open 20 clients concurrently,
    # assert every client gets its own payload back.
    pass
