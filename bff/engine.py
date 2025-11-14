import time
import threading
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from bff.settings import settings
import stream_pb2

@dataclass
class AsrResult:
    start_s: float
    end_s: float
    text: str
    is_final: bool

class RealtimeEngine:
    """Minimal demo engine. Swap 'transcribe_window' with your diar-first pipeline."""
    def __init__(self):
        self.lock = threading.Lock()
        self.buffers = {}         # session_id -> np.float32 samples
        self.last_emit_t = {}     # session_id -> float (seconds)
        self.sample_rate = settings.sample_rate
        self.max_buf = int(settings.max_buffer_seconds * self.sample_rate)

    def feed(self, session_id: str, pcm16: bytes) -> List[AsrResult]:
        """Feed raw PCM16 mono; return zero or more results to push immediately."""
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        now = time.monotonic()
        with self.lock:
            buf = self.buffers.get(session_id, np.zeros(0, dtype=np.float32))
            buf = np.concatenate([buf, x])
            if buf.size > self.max_buf:
                buf = buf[-self.max_buf:]
            self.buffers[session_id] = buf
            last_emit = self.last_emit_t.get(session_id, 0.0)

        results: List[AsrResult] = []

        # --- BEGIN: demo logic (replace with your windowing + diar + ASR) ---
        # Every ~0.6s emit a "partial", every ~3s emit a "final"
        dur = buf.size / self.sample_rate
        if now - last_emit > 0.6 and dur > 0.6:
            text = f"[partial dur={dur:.1f}s]"  # replace with real text
            results.append(AsrResult(start_s=max(0.0, dur-2.0),
                                     end_s=dur,
                                     text=text,
                                     is_final=False))
            self.last_emit_t[session_id] = now
        if int(dur) % 3 == 0 and dur > 0 and (now - last_emit) > 1.0:
            text = f"[final up to {dur:.1f}s]"   # replace with real text
            results.append(AsrResult(start_s=0.0, end_s=dur, text=text, is_final=True))
            self.last_emit_t[session_id] = now
        # --- END: demo logic ---

        return results

    def to_asr_events(self, session_id: str, r: AsrResult) -> stream_pb2.AsrEvent:
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=r.start_s,
            end_s=r.end_s,
            text=r.text,
            type=stream_pb2.PARTIAL if not r.is_final else stream_pb2.FINAL,
            lang="",  # optional
        )
