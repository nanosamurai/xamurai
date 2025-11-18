from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # WebSocket frame assumptions
    sample_rate: int = 16000
    frame_samples: int = 320          # e.g., 20 ms @16k, but client can send any even size
    max_buffer_seconds: float = 5.0   # in-memory ring buffer for realtime engine

    # Kafka
    kafka_bootstrap: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP")
    kafka_topic_audio: str = Field("audio.raw", env="KAFKA_TOPIC_AUDIO")
    kafka_topic_refined: str = Field("transcripts.refined", env="KAFKA_TOPIC_REFINED")
    kafka_client_id: str = Field("bff", env="KAFKA_CLIENT_ID")
    kafka_acks: str = Field("1", env="KAFKA_ACKS")
    kafka_compression: str = Field("zstd", env="KAFKA_COMPRESSION")
    kafka_linger_ms: int = Field(5, env="KAFKA_LINGER_MS")
    kafka_batch_size: int = Field(131072, env="KAFKA_BATCH_SIZE")  # 128k

settings = Settings()
