import threading

import pytest

from xamurai_serving import SessionSlots, max_sessions_from_env, serving_instance_id


def test_session_slots_admit_only_the_configured_number():
    slots = SessionSlots(2)

    assert slots.acquire() is True
    assert slots.acquire() is True
    assert slots.acquire() is False
    assert slots.active == 2

    slots.release()
    assert slots.acquire() is True


def test_session_slot_acquisition_is_atomic():
    slots = SessionSlots(1)
    barrier = threading.Barrier(8)
    results = []

    def contend():
        barrier.wait()
        results.append(slots.acquire())

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == 7


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_max_sessions_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("RT_SERVING_MAX_SESSIONS", value)

    with pytest.raises(ValueError, match="positive integer"):
        max_sessions_from_env()


def test_serving_instance_id_prefers_explicit_configuration(monkeypatch):
    monkeypatch.setenv("RT_SERVING_INSTANCE_ID", "replica-a")

    assert serving_instance_id() == "replica-a"
