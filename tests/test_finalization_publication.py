"""The shared and standalone finalizers retain the same publication contract."""
import pytest
from proto_gen import stream_pb2 as pb
from xamurai_serving import finalization as worker


class Message:
    def __init__(self, tracks="whisperx", retained=False):
        self.tracks = tracks
        self.retained = retained
    def headers(self):
        return [("x-final-tracks", self.tracks.encode()),
                ("x-store-recording", str(self.retained).lower().encode())]
    def value(self):
        return pb.RecordingFinished(session_id="session", tenant_id="tenant",
                                    recording_url="file:///recording.wav", lang="en").SerializeToString()


def test_output_ack_precedes_deferred_deletion_and_preserves_words(monkeypatch):
    published = []
    monkeypatch.setattr(worker, "TRACK_ID", "whisperx")
    monkeypatch.setattr(worker, "_produce_with_ack", lambda _p, **kw: published.append(kw))
    monkeypatch.setattr(worker, "_delete_recording_url", lambda _url: pytest.fail("early deletion"))
    def transcribe(_path, *, tenant, lang):
        assert (tenant, lang) == ("tenant", "en")
        return "hello", [dict(start_s=0, end_s=1, text="hello", speaker="speaker",
                              words=[dict(start_s=.1, end_s=.9, text="hello")])]
    delete_url = worker.process_message(Message(), producer=None, transcribe=transcribe, model="medium")
    assert delete_url == "file:///recording.wav"
    result = pb.SessionTranscript.FromString(published[0]["value"])
    assert (result.track_id, result.tenant_id, result.segments[0].words[0].text) == ("whisperx", "tenant", "hello")
    assert published[0]["key"] == b"session"


def test_unselected_track_skips_inference_and_returns_no_deletion(monkeypatch):
    monkeypatch.setattr(worker, "TRACK_ID", "whisperx-shared")
    assert worker.process_message(Message(), producer=None,
                                  transcribe=lambda *_a, **_k: pytest.fail("unselected inference"),
                                  model="medium") is None


def test_failed_publication_never_returns_cleanup(monkeypatch):
    monkeypatch.setattr(worker, "TRACK_ID", "whisperx")
    def fail(*_args, **_kwargs): raise RuntimeError("no ack")
    monkeypatch.setattr(worker, "_produce_with_ack", fail)
    with pytest.raises(RuntimeError):
        worker.process_message(Message(), producer=None, transcribe=lambda *_a, **_k: ("", []), model="medium")
