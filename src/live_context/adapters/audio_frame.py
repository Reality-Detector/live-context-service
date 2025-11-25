# src/live_context/adapters/audio_frame.py
from dataclasses import dataclass
from typing import Literal
import time

AudioEncoding = Literal["pcm16"]


@dataclass
class AudioFrame:
    """
    Canonical audio frame passed from adapters into the ASR pipeline.
    All adapters should convert their platform-specific audio into this shape.
    """
    t_capture: float       # timestamp in seconds (monotonic or wall-clock)
    sample_rate: int       # e.g. 16000
    num_channels: int      # e.g. 1
    encoding: AudioEncoding
    data: bytes            # raw PCM16 little-endian audio

    @staticmethod
    def now(sample_rate: int, num_channels: int, data: bytes):
        return AudioFrame(
            t_capture=time.time(),
            sample_rate=sample_rate,
            num_channels=num_channels,
            encoding="pcm16",
            data=data,
        )
