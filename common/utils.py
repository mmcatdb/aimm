from typing import Any, Protocol
import dataclasses, json
from contextlib import contextmanager

# Add small epsilon to avoid division by zero
EPSILON = 1e-8

class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            dataclass: Any = o
            return dataclasses.asdict(dataclass)
        return super().default(o)

class Closeable(Protocol):
    def close(self) -> None:
        pass

@contextmanager
def auto_close(closeable: Closeable):
    try:
        yield closeable
    except Exception:
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        closeable.close()
