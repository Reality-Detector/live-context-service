# src/live_context/core/session.py
from typing import Any, Dict, Optional
from .event_bus import EventBus
from ..pipelines.asr_pipeline import ASRPipeline
from ..providers.base_asr import ASRProvider
from ..protocol import events_v1


class Session:
    """
    Represents a single live context session.
    For now: ASR-only.
    """

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.event_bus = EventBus()
        self._asr_pipeline: Optional[ASRPipeline] = None
        self._active = False

    def attach_asr_provider(self, provider: ASRProvider) -> None:
        """
        Attach an ASR provider and build the ASR pipeline around it.
        """
        self._asr_pipeline = ASRPipeline(
            session_id=self.session_id,
            event_bus=self.event_bus,
            provider=provider,
        )

    @property
    def asr_pipeline(self) -> ASRPipeline:
        assert self._asr_pipeline is not None, "ASR provider not attached"
        return self._asr_pipeline

    async def start(self) -> None:
        self._active = True
        self.event_bus.emit(events_v1.session_started(self.config))
        # Do NOT call provider.start() here; Deepgram will connect on first audio frame.


    async def stop(self) -> None:
        if self._asr_pipeline:
            await self._asr_pipeline.stop()
        self._active = False
        self.event_bus.emit(events_v1.session_stopped())
