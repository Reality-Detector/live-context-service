# src/live_context/providers/deepgram_asr.py
import os
import json
import asyncio
import websockets
from typing import Any, Optional, List, Dict

from .base_asr import ASRProvider
from ..adapters.audio_frame import AudioFrame
from ..protocol import events_v1
from ..core.quality import compute_quality

# Explicitly describe audio format & options to Deepgram
DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=linear16"
    "&sample_rate=48000"
    "&punctuate=true"
    "&interim_results=true"
    "&diarize=true"
)


class DeepgramASRProvider(ASRProvider):
    """
    Deepgram realtime ASR provider with:
    - lazy connection on first audio frame
    - stable utterance IDs across partials
    - quality metrics added to each event
    """

    def __init__(self, session_id: str):
        super(DeepgramASRProvider, self).__init__()
        self.session_id = session_id

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._loop = asyncio.get_event_loop()
        self._receiver_task: Optional[asyncio.Task] = None

        # Utterance tracking
        self._next_utt_seq: int = 1
        self._current_utt_id: Optional[str] = None

    # ---------- Connection management ----------

    async def _connect(self) -> None:
        if self._ws is not None:
            return

        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        headers = {
            "Authorization": "Token {}".format(api_key),
        }

        self._ws = await websockets.connect(DEEPGRAM_URL, extra_headers=headers)
        self._receiver_task = self._loop.create_task(self._receive_loop())
        print("Deepgram connection established")

    async def start(self) -> None:
        """
        We keep this for API consistency, but rely on lazy connect in push_frame().
        """
        return

    async def stop(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            self._receiver_task = None
        print("Deepgram connection closed")

    # ---------- Audio input ----------

    async def push_frame(self, frame: AudioFrame) -> None:
        """
        Called by the ASR pipeline for each incoming AudioFrame.
        """
        if self._ws is None:
            await self._connect()

        try:
            await self._ws.send(frame.data)
        except websockets.ConnectionClosed:
            # Try a simple reconnect once
            self._ws = None
            await self._connect()
            await self._ws.send(frame.data)

    # ---------- Receiving results ----------

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                await self._handle_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("Deepgram receive loop error:", e)

    async def _handle_message(self, msg: Any) -> None:
        if self._on_event is None:
            return

        try:
            data = json.loads(msg)
        except Exception:
            return

        if data.get("type") != "Results":
            return

        channel = data.get("channel", {})
        alts: List[Dict[str, Any]] = channel.get("alternatives", [])
        if not alts:
            return

        alt = alts[0]
        text: str = alt.get("transcript", "") or ""
        if not text.strip():
            return

        confidence = float(alt.get("confidence") or 0.0)
        is_final: bool = bool(channel.get("is_final", False))

        # Extract timing from words if available
        words = alt.get("words") or []
        if words:
            try:
                t_start = float(words[0].get("start", 0.0))
                t_end = float(words[-1].get("end", 0.0))
            except Exception:
                t_start = t_end = 0.0
        else:
            t_start = t_end = 0.0

        # Simple diarization placeholder – Deepgram can provide speaker info;
        # for now we just keep UNKNOWN and will refine later.
        speaker = "UNKNOWN"

        # ---------- Utterance ID handling ----------
        # Reuse a single utterance ID across all partials until we get a final.
        if self._current_utt_id is None:
            self._current_utt_id = "utt_{}".format(self._next_utt_seq)
            self._next_utt_seq += 1

        utt_id = self._current_utt_id

        # ---------- Quality metrics ----------
        quality = compute_quality(text=text, confidence=confidence, is_final=is_final)

        if not is_final:
            event = events_v1.partial_utterance(
                session_id=self.session_id,
                utterance_id=utt_id,
                speaker=speaker,
                t_start=t_start,
                t_end=t_end,
                text=text,
                confidence=confidence,
            )
            event["quality"] = quality
            self._on_event(event)
        else:
            event = events_v1.final_utterance(
                session_id=self.session_id,
                utterance_id=utt_id,
                speaker=speaker,
                t_start=t_start,
                t_end=t_end,
                raw_text=text,
                confidence=confidence,
            )
            event["quality"] = quality
            self._on_event(event)

            # Current utterance is now complete; next result starts a new one.
            self._current_utt_id = None
