# src/live_context/providers/base_asr.py
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional
from ..adapters.audio_frame import AudioFrame

ASREventCallback = Callable[[Dict[str, Any]], None]


class ASRProvider(ABC):
    """
    Abstract ASR provider.
    Implementations receive AudioFrame objects and emit events via `_on_event`.
    """

    def __init__(self):
        self._on_event: Optional[ASREventCallback] = None

    def set_event_callback(self, cb: ASREventCallback) -> None:
        self._on_event = cb

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def push_frame(self, frame: AudioFrame) -> None:
        ...
