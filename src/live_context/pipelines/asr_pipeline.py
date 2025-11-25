# src/live_context/pipelines/asr_pipeline.py
from typing import Dict, Any
from ..core.event_bus import EventBus
from ..adapters.audio_frame import AudioFrame
from ..providers.base_asr import ASRProvider


class ASRPipeline:
    """
    Orchestrates: AudioFrame -> ASRProvider -> events on EventBus.
    """

    def __init__(
        self,
        session_id: str,
        event_bus: EventBus,
        provider: ASRProvider,
    ):
        self.session_id = session_id
        self.event_bus = event_bus
        self.provider = provider

        # Wire provider callback to our handler:
        self.provider.set_event_callback(self.handle_provider_event)

    async def start(self) -> None:
        await self.provider.start()

    async def stop(self) -> None:
        await self.provider.stop()

    async def push_frame(self, frame: AudioFrame) -> None:
        await self.provider.push_frame(frame)

    def handle_provider_event(self, event: Dict[str, Any]) -> None:
        """
        Provider emits already-in-v1-shape events; we just forward.
        Later we can normalize or enrich here.
        """
        self.event_bus.emit(event)
