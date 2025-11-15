import os
import tempfile
import time
from collections import defaultdict, deque
from typing import Deque, Tuple

import numpy as np
import soundfile as sf
from confluent_kafka import Consumer, Producer, KafkaException

import stream_pb2

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_REFINED = os.getenv("KAFKA_TOPIC_REFINED", "transcripts.refined")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "whisperx-offline")

SLICE_SECONDS = 60.0
SR = 16000

def make_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.partition.fetch.bytes": 5_000_000,
        "fetch.wait.max.ms": 50,
    })

def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "client.id": "whisperx-offline",
        "compression.type": "zstd",
        "linger.ms": 10,
        "batch.size": 131072
    })

def run_whisperx(wav_path: str) -> Tuple[str, list]:
    """
    TODO: Implement real WhisperX call.
    Return text and list of (start_s, end_s, text, speaker_opt).
    For MVP, we return the duration as text.
    """
    import soundfile as sf
    _, sr = sf.read(wav_path)
    dur = sf.info(wav_path).duration
    return f"[whisperx result ~{dur:.1f}s]", [(0.0, dur, "[offline text]", "")]

def main():
    c = make_consumer()
    p = make_producer()
    c.subscribe([TOPIC_AUDIO])

    # Per-session rolling PCM16 mono buffer
    buffers: dict[str, Deque[np.ndarray]] = defaultdict(deque)
    buf_samples: dict[str, int] = defaultdict(int)

    try:
        while True:
            msg = c.poll(0.1)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            key = msg.key().decode("utf-8") if msg.key() else ""
            audio = stream_pb2.AudioChunk()
            audio.ParseFromString(msg.value())

            # Append to session buffer
            arr = np.frombuffer(audio.pcm16_le, dtype="<i2")
            buffers[audio.session_id].append(arr)
            buf_samples[audio.session_id] += arr.size

            # If we have ≥ SLICE_SECONDS, dump to WAV and process
            if buf_samples[audio.session_id] >= int(SLICE_SECONDS * SR):
                # Gather exactly SLICE_SECONDS
                need = int(SLICE_SECONDS * SR)
                parts = []
                while need > 0 and buffers[audio.session_id]:
                    ch = buffers[audio.session_id][0]
                    if ch.size <= need:
                        parts.append(buffers[audio.session_id].popleft())
                        need -= ch.size
                        buf_samples[audio.session_id] -= ch.size
                    else:
                        parts.append(ch[:need])
                        buffers[audio.session_id][0] = ch[need:]
                        buf_samples[audio.session_id] -= need
                        need = 0

                pcm = np.concatenate(parts).astype("<i2")
                # Write temp WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name
                # Convert to float for sf.write
                x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                sf.write(wav_path, x, SR, subtype="PCM_16")

                # WhisperX inference
                text, segments = run_whisperx(wav_path)

                # Publish RefinedEvent(s)
                for (s0, s1, seg_text, speaker) in segments:
                    ev = stream_pb2.RefinedEvent(
                        session_id=audio.session_id,
                        start_s=s0,
                        end_s=s1,
                        text=seg_text,
                        speaker=speaker,
                        supersedes_seq=[],
                    )
                    p.produce(
                        topic=TOPIC_REFINED,
                        key=audio.session_id.encode("utf-8"),
                        value=ev.SerializeToString(),
                    )
                p.poll(0)

                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

            # Commit offset (you can switch to batched commits)
            c.commit(msg, asynchronous=True)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            c.close()
        except Exception:
            pass
        try:
            p.flush(2.0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
