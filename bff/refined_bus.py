import asyncio
import logging
import threading
from typing import Dict, Optional

from confluent_kafka import Consumer, KafkaException
from proto import stream_pb2

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_REFINED = "transcripts.refined"
GROUP_ID = "bff-refined-1"


class RefinedBus:
    """
    Single Kafka consumer for transcripts.refined.
    Distributes messages into per-session asyncio.Queues.
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._consumer: Optional[Consumer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # session_id -> asyncio.Queue
        self._queues: Dict[str, asyncio.Queue] = {}

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start background thread; must be called from FastAPI startup with running loop."""
        if self._running:
            return
        self._running = True
        self._loop = loop

        logger.info("Starting RefinedBus Kafka consumer")

        self._consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": GROUP_ID,
            "enable.auto.commit": True,
            "auto.offset.reset": "latest",
        })
        self._consumer.subscribe([TOPIC_REFINED])

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:
                pass

    async def register(self, session_id: str) -> asyncio.Queue:
        """
        Called by WS handler when a new client for session_id connects.
        Returns a queue where RefinedEvents for that session will appear.
        """
        q = asyncio.Queue()
        self._queues[session_id] = q
        logger.info("Registered refined queue for session_id=%s", session_id)
        return q

    def unregister(self, session_id: str):
        self._queues.pop(session_id, None)
        logger.info("Unregistered refined queue for session_id=%s", session_id)

    def _run(self):
        """Kafka polling loop (runs in background thread)."""
        assert self._consumer is not None
        while self._running:
            try:
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error("Kafka refined error: %s", msg.error())
                    continue

                key = msg.key().decode("utf-8") if msg.key() else ""
                q = self._queues.get(key)
                if not q:
                    # No active WS for that session; skip (or buffer/DB if you want persistence)
                    continue

                ev_pb = stream_pb2.RefinedEvent()
                ev_pb.ParseFromString(msg.value())

                payload = {
                    "type": "refined",
                    "session_id": ev_pb.session_id,
                    "start_s": ev_pb.start_s,
                    "end_s": ev_pb.end_s,
                    "text": ev_pb.text,
                    "speaker": ev_pb.speaker,
                    "supersedes_seq": list(ev_pb.supersedes_seq),
                }

                # queue.put is async → schedule on main loop
                asyncio.run_coroutine_threadsafe(q.put(payload), self._loop)

            except KafkaException as e:
                logger.exception("KafkaException in RefinedBus: %s", e)
            except Exception:
                logger.exception("Unexpected error in RefinedBus._run")

        logger.info("RefinedBus stopped.")
