"""Thread-safe current audio endpoint tracking; no dependency on Tkinter/Whisper."""
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioEndpoint:
    frames: int = 0
    captured_at: float = 0.0


class AudioClock:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = AudioEndpoint()

    def update(self, frames: int, captured_at: float) -> None:
        with self._lock:
            if frames < self._latest.frames:
                raise ValueError('Audio endpoint cannot move backward')
            self._latest = AudioEndpoint(frames, captured_at)

    def snapshot(self) -> AudioEndpoint:
        with self._lock:
            return self._latest


audio_clock = AudioClock()
