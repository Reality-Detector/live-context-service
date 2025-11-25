# src/live_context/utils/wav_reader.py
import wave
from typing import Generator
from ..adapters.audio_frame import AudioFrame


def stream_wav_as_frames(path: str, chunk_ms: int = 100) -> Generator[AudioFrame, None, None]:
    """
    Yield AudioFrame chunks from a 16-bit PCM WAV file.
    """
    wf = wave.open(path, "rb")
    try:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError("Expecting 16-bit PCM WAV, got sampwidth={}".format(sampwidth))

        frames_per_chunk = int(sample_rate * chunk_ms / 1000.0)

        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            yield AudioFrame.now(sample_rate=sample_rate, num_channels=num_channels, data=data)
    finally:
        wf.close()
