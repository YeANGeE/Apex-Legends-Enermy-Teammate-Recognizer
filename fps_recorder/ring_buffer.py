from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


class TimestampRing(Generic[T]):
    """Thread-safe, time-bounded rolling buffer."""

    def __init__(
        self, duration_ns: int, timestamp_getter, max_items: int | None = None
    ) -> None:
        self.duration_ns = int(duration_ns)
        self._timestamp_getter = timestamp_getter
        self.max_items = max_items
        self._items: deque[T] = deque()
        self._lock = Lock()

    def append(self, item: T) -> None:
        timestamp = int(self._timestamp_getter(item))
        cutoff = timestamp - self.duration_ns
        with self._lock:
            self._items.append(item)
            while self._items and int(self._timestamp_getter(self._items[0])) < cutoff:
                self._items.popleft()
            while self.max_items is not None and len(self._items) > self.max_items:
                self._items.popleft()

    def snapshot_since(self, timestamp_ns: int) -> list[T]:
        with self._lock:
            return [
                item
                for item in self._items
                if int(self._timestamp_getter(item)) >= timestamp_ns
            ]

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def snapshot_with_predecessor(self, timestamp_ns: int) -> list[T]:
        """Include one causal seed before the clip boundary, never a future seed."""
        with self._lock:
            earlier = [row for row in self._items
                       if int(self._timestamp_getter(row)) < timestamp_ns]
            later = [row for row in self._items
                     if int(self._timestamp_getter(row)) >= timestamp_ns]
            return earlier[-1:] + later

    def span_ns(self) -> int:
        with self._lock:
            if len(self._items) < 2:
                return 0
            return int(self._timestamp_getter(self._items[-1])) - int(
                self._timestamp_getter(self._items[0])
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
