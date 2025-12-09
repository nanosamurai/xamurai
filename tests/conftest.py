# tests/conftest.py

import sys
from pathlib import Path

# Ensure project root (the directory that contains rtservice/, bff/, etc.) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import grpc

from rtservice.server import create_realtime_asr_server


@pytest.fixture(scope="session")
def grpc_server_address():
    """
    Start the RealtimeASR gRPC server in-process for all tests in the session.
    Yields the address to connect to (host:port).
    """
    port = 50052  # use a test port, not 50051
    server = create_realtime_asr_server(port=port)

    server.start()
    addr = f"localhost:{port}"
    print(f"[test] RealtimeASR server started at {addr}")

    try:
        yield addr
    finally:
        # Graceful shutdown
        server.stop(grace=0).wait()
        print("[test] RealtimeASR server stopped")


@pytest.fixture
def grpc_channel(grpc_server_address):
    """
    Create a gRPC client channel to the test server.
    New channel per test (scope='function').
    """
    channel = grpc.insecure_channel(grpc_server_address)
    grpc.channel_ready_future(channel).result(timeout=10)
    try:
        yield channel
    finally:
        channel.close()
