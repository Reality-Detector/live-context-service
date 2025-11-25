# src/live_context/core/event_bus.py
from typing import Callable, Dict, Any, List

Event = Dict[str, Any]
Subscriber = Callable[[Event], None]


class EventBus:
    """
    Simple in-process pub/sub bus.
    Pipelines emit events here, subscribers (UI, LangChain, logs) listen.
    """
    def __init__(self):
        self._subscribers: List[Subscriber] = []

    def subscribe(self, fn: Subscriber) -> None:
        self._subscribers.append(fn)

    def emit(self, event: Event) -> None:
        for fn in list(self._subscribers):
            fn(event)
