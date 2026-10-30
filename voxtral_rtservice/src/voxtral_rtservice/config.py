"""Operator-only settings, validated before importing the GPU runtime."""

from dataclasses import dataclass
import math
import os

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 2_560  # 160 ms; two native Voxtral decoding steps.
FRAME_BYTES = FRAME_SAMPLES * 2
MAX_CHUNK_BYTES = 32_000  # One second of PCM16; protobuf has a separate limit.
VLLM_VERSION = "0.32.0"


@dataclass(frozen=True)
class Config:
    maximum_sessions: int
    epoch_seconds: int = 60
    max_model_len: int = 2048
    kv_cache_mib: int = 2048
    max_batched_tokens: int = 2048
    tensor_parallel_size: int = 1
    audio_queue_frames: int = 32
    idle_timeout: float = 30.0
    progress_timeout: float = 60.0
    output_timeout: float = 30.0
    cleanup_timeout: float = 10.0
    max_epoch_text_bytes: int = 65_536
    bind_address: str = "127.0.0.1"
    port: int = 50052

    def __post_init__(self):
        for name, low, high in (
            ("maximum_sessions", 1, 1024), ("epoch_seconds", 10, 600),
            ("max_model_len", 1024, 131072), ("kv_cache_mib", 256, 1_048_576),
            ("max_batched_tokens", 128, 131072), ("tensor_parallel_size", 1, 64),
            ("audio_queue_frames", 1, 256), ("max_epoch_text_bytes", 1024, 1_048_576),
            ("port", 0, 65535),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name in ("idle_timeout", "progress_timeout", "output_timeout", "cleanup_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 600:
                raise ValueError(f"{name} must be finite and in (0, 600]")
        # 12.5 decoding steps per second, plus prefix/padding and safety margin.
        if self.max_model_len < math.ceil(self.epoch_seconds / 0.08) + 512:
            raise ValueError("max_model_len cannot cover the configured epoch")
        if self.max_batched_tokens < self.maximum_sessions:
            raise ValueError("max_batched_tokens must cover maximum_sessions")
        if not self.bind_address or any(c.isspace() for c in self.bind_address):
            raise ValueError("bind_address must be nonempty without whitespace")

    @classmethod
    def from_env(cls):
        # Capacity is a deployment decision, never an implicit one-session cap.
        if "RT_SERVING_MAX_SESSIONS" not in os.environ:
            raise ValueError("Set RT_SERVING_MAX_SESSIONS to the GPU-qualified replica capacity")
        return cls(
            maximum_sessions=int(os.environ["RT_SERVING_MAX_SESSIONS"]),
            epoch_seconds=int(os.getenv("VOXTRAL_EPOCH_SECONDS", "60")),
            max_model_len=int(os.getenv("VOXTRAL_MAX_MODEL_LEN", "2048")),
            kv_cache_mib=int(os.getenv("VOXTRAL_KV_CACHE_MIB", "2048")),
            max_batched_tokens=int(os.getenv("VOXTRAL_MAX_BATCHED_TOKENS", "2048")),
            tensor_parallel_size=int(os.getenv("VOXTRAL_TENSOR_PARALLEL_SIZE", "1")),
            audio_queue_frames=int(os.getenv("VOXTRAL_AUDIO_QUEUE_FRAMES", "32")),
            idle_timeout=float(os.getenv("VOXTRAL_IDLE_TIMEOUT_SECONDS", "30")),
            progress_timeout=float(os.getenv("VOXTRAL_PROGRESS_TIMEOUT_SECONDS", "60")),
            output_timeout=float(os.getenv("VOXTRAL_OUTPUT_TIMEOUT_SECONDS", "30")),
            cleanup_timeout=float(os.getenv("VOXTRAL_CLEANUP_TIMEOUT_SECONDS", "10")),
            max_epoch_text_bytes=int(os.getenv("VOXTRAL_MAX_EPOCH_TEXT_BYTES", "65536")),
            bind_address=os.getenv("VOXTRAL_RTSERVICE_BIND_ADDR", "127.0.0.1"),
            port=int(os.getenv("VOXTRAL_RTSERVICE_PORT", "50052")),
        )
