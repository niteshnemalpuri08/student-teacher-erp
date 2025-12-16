import json
import time
import queue
from threading import Lock

# Simple in-memory SSE broadcaster suitable for single-process dev POC.
# Each subscriber gets a Queue; publish() puts messages on all subscriber queues.

_subscribers = []
_lock = Lock()


def subscribe():
    q = queue.Queue()
    with _lock:
        _subscribers.append(q)

    def generator(timeout=15):
        try:
            # Send a welcome event
            yield 'event: welcome\n'
            yield f'data: {json.dumps({"message":"connected"})}\n\n'

            while True:
                try:
                    msg = q.get(timeout=timeout)
                    # msg should be a JSON-serializable dict
                    payload = json.dumps(msg)
                    yield f'data: {payload}\n\n'
                except queue.Empty:
                    # Keepalive comment to keep connection alive
                    yield ': keep-alive\n\n'
        finally:
            # unsubscribe
            with _lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return generator


def publish(message):
    # message should be a JSON-serializable dict
    with _lock:
        for q in list(_subscribers):
            try:
                q.put_nowait(message)
            except Exception:
                # Ignore and continue
                pass
