from drsynth_common.stream_controls import parse_stream_controls_from_kafka_headers


def test_parse_stream_controls_defaults_are_backwards_compatible():
    c = parse_stream_controls_from_kafka_headers(None)
    assert c.want_realtime is True
    assert c.want_refined is True
    assert c.want_final is True
    assert c.store_recording is True


def test_parse_stream_controls_outputs_subset_and_store_recording_false():
    headers = [
        ("x-outputs", b"realtime,final"),
        ("x-store-recording", b"false"),
    ]
    c = parse_stream_controls_from_kafka_headers(headers)
    assert c.want_realtime is True
    assert c.want_refined is False
    assert c.want_final is True
    assert c.store_recording is False


def test_parse_stream_controls_outputs_empty_means_none():
    headers = [("x-outputs", b"   ")]
    c = parse_stream_controls_from_kafka_headers(headers)
    assert c.want_realtime is False
    assert c.want_refined is False
    assert c.want_final is False
