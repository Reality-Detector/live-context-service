# src/live_context/providers/deepgram_asr.py
import os
import json
import asyncio
import websockets
from typing import Any, Optional, List, Dict, Tuple

from .base_asr import ASRProvider
from ..adapters.audio_frame import AudioFrame
from ..protocol import events_v1
from ..core.quality import compute_quality

# Explicitly describe audio format & options to Deepgram.
# We enable:
# - interim_results=true  → partial hypotheses while user is speaking
# - endpointing           → VAD-based silence detection to emit periodic finals
# - utterance_end_ms      → gap-based utterance boundaries for more semantic chunks
DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=linear16"
    "&sample_rate=48000"
    "&channels=2"
    "&multichannel=true"
    # Model and formatting
    "&model=nova-3-general"
    "&punctuate=true"
    "&interim_results=true"
    # Diarization stays enabled for future use
    "&diarize=true"
    # Endpointing / segmentation:
    #   - endpointing: ms of silence before Deepgram finalizes a segment
    #   - utterance_end_ms: ms gap between recognized words to trigger UtteranceEnd
    # These values can be tuned, but are a reasonable starting point for
    # sentence-like segments in typical speech or video audio.
    "&endpointing=750"
    "&utterance_end_ms=1000"
    "&vad_events=true"
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
        # We maintain a monotonically increasing sequence counter and track the
        # "open" utterance per speaker so that:
        #   - each utterance_id is tied to exactly one speaker, and
        #   - each speaker has at most one active utterance receiving partials.
        self._next_utt_seq: int = 1
        self._current_utt_by_speaker: Dict[str, str] = {}

        # Speaker smoothing: track the last stable speaker we emitted so we can
        # avoid dropping to UNKNOWN or rapidly flapping between speakers when
        # Deepgram's diarization is uncertain. Threshold is in seconds.
        self._last_stable_speaker: Optional[str] = None
        self._speaker_min_duration: float = 0.15

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

    def _infer_speaker(self, words: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Infer a stable speaker label (e.g. "spk_0", "spk_1") from Deepgram's
        diarized word-level metadata.
        Returns a tuple of (speaker_label, total_word_duration_for_label).
        """
        if not words:
            return "UNKNOWN", 0.0

        durations: Dict[str, float] = {}
        for w in words:
            sp = w.get("speaker")
            if sp is None:
                continue

            try:
                idx = int(sp)
            except Exception:
                continue

            key = f"spk_{idx}"
            try:
                start = float(w.get("start", 0.0))
                end = float(w.get("end", 0.0))
                dur = end - start
                if dur <= 0:
                    dur = 0.01
            except Exception:
                dur = 0.01

            durations[key] = durations.get(key, 0.0) + dur

        if not durations:
            return "UNKNOWN", 0.0

        # Pick the speaker that dominates this utterance by total word duration.
        label, dur = max(durations.items(), key=lambda kv: kv[1])
        return label, float(dur)

    def _stabilize_speaker_label(self, speaker: str, duration: float) -> str:
        """
        Apply simple smoothing so extremely short or missing diarization segments
        fall back to the previously stable speaker instead of creating flicker.
        """
        if speaker == "UNKNOWN":
            return self._last_stable_speaker or "UNKNOWN"

        if duration < self._speaker_min_duration and self._last_stable_speaker:
            # Treat very short, low-confidence diarization as likely belonging to
            # the previously active speaker.
            return self._last_stable_speaker

        self._last_stable_speaker = speaker
        return speaker

    def _extract_channel_index(
        self,
        data: Dict[str, Any],
        channel: Dict[str, Any],
        alt: Dict[str, Any],
        words: List[Dict[str, Any]],
    ) -> Optional[int]:
        """
        Try to pull a stable channel index from Deepgram responses.
        DG may surface this at different keys depending on version.
        """
        candidates = [
            data.get("channel_index"),
            channel.get("channel_index"),
            channel.get("channel"),
            alt.get("channel_index"),
            alt.get("channel"),
            data.get("metadata", {}).get("channel_index"),
            data.get("metadata", {}).get("channel"),
        ]

        # Check words if present (first non-null wins)
        if words:
            w0 = words[0]
            candidates.append(w0.get("channel_index"))
            candidates.append(w0.get("channel"))

        idx = next((c for c in candidates if c is not None), None)

        try:
            return int(idx)
        except Exception:
            return None

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

        # Deepgram v1 listen results expose finality at the top level as
        # `is_final` / `speech_final`. Older examples sometimes showed this
        # on `channel`, so we OR them together for robustness.
        is_final: bool = bool(
            data.get("is_final")
            or data.get("speech_final")
            or channel.get("is_final", False)
        )

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

        # ---------- Speaker & utterance ID handling ----------
        # Channel-based mapping:
        #   ch 0 (mic)     -> always "user" (no diarization)
        #   ch 1 (system)  -> keep diarization within system channel (system:spk_N)
        # Fallback: if channel metadata is missing, use legacy diarization.
        channel_index = self._extract_channel_index(data, channel, alt, words)

        if channel_index == 0:
            speaker = "user"
        elif channel_index == 1:
            sp_label, sp_dur = self._infer_speaker(words)
            sp_label = self._stabilize_speaker_label(sp_label, sp_dur)
            speaker = f"system:{sp_label}" if sp_label not in (None, "UNKNOWN") else "system"
        else:
            # Legacy path: single channel or no channel metadata
            sp_label, sp_dur = self._infer_speaker(words)
            speaker = self._stabilize_speaker_label(sp_label, sp_dur)

        utt_id = self._current_utt_by_speaker.get(speaker)
        if utt_id is None:
            # Incorporate speaker in the utterance_id to make it easy to debug
            # and to keep IDs unique across speakers.
            base = speaker if speaker != "UNKNOWN" else "utt"
            utt_id = f"{base}_{self._next_utt_seq}"
            self._next_utt_seq += 1
            self._current_utt_by_speaker[speaker] = utt_id

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

            # Current utterance for this speaker is now complete; next result
            # from the same speaker will start a new utterance.
            self._current_utt_by_speaker.pop(speaker, None)
