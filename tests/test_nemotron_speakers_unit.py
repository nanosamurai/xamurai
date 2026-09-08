import io
import json
from dataclasses import replace

import numpy as np
import pytest

from nemotron_rtservice.native import SpeakerWord, diarization_from_env
from nemotron_rtservice.speakers import EnrolledSpeakers, speaker_turns


def test_turns_preserve_punctuation_and_split_only_at_speaker_changes():
    text = 'Hello, world!  Yes, I agree.'
    words = tuple(SpeakerWord(word, i, i + 0.8, 1 if i < 2 else 2)
                  for i, word in enumerate(text.split()))
    turns = speaker_turns(text, words, 0, 6)
    assert ''.join(turn.text for turn in turns) == text
    assert [(turn.speaker, turn.start_s, turn.end_s) for turn in turns] == [(1, 0, 1.8), (2, 2, 4.8)]


@pytest.mark.parametrize('changes', [dict(text='missing'), dict(start_s=float('nan')),
                                     dict(end_s=-1), dict(speaker=5), dict(start_s=-1),
                                     dict(start_s=3, end_s=4)])
def test_invalid_words_fall_back_without_dropping_text(changes):
    words = (replace(SpeakerWord('Hello.', 0, 1, 1), **changes),)
    assert speaker_turns('Hello.', words, 0, 2) == ()


def test_incomplete_native_text_coverage_falls_back():
    assert speaker_turns('one missing two', (SpeakerWord('one', 0, 1, 1), SpeakerWord('two', 2, 3, 2)), 0, 3) == ()


def test_unknown_speaker_is_not_promoted_to_a_known_slot():
    assert speaker_turns('Hello.', (SpeakerWord('Hello.', 0, 1, 0),), 0, 2)[0].speaker == 0


def test_late_word_end_does_not_include_the_next_speaker_audio():
    turns = speaker_turns('One. Two.', (SpeakerWord('One.', 0, 4, 1), SpeakerWord('Two.', 2, 5, 2)), 0, 5)
    assert turns[0].end_s == turns[1].start_s == 2


@pytest.mark.parametrize('word_end', [30.4, 100])
@pytest.mark.parametrize('word_start', [29.6, 30.16])
def test_native_word_lookahead_is_clipped_to_the_final_audio_boundary(word_start, word_end):
    text = 'A long utterance ends.'
    words = (SpeakerWord('A', 0.4, 0.6, 1),
             SpeakerWord('long', 1, 2, 1),
             SpeakerWord('utterance', 2.4, 28, 1),
             SpeakerWord('ends.', word_start, word_end, 1))
    turns = speaker_turns(text, words, 0, 30.02)
    assert len(turns) == 1
    assert turns[0].text == text
    assert (turns[0].speaker, turns[0].start_s, turns[0].end_s) == (1, 0.4, 30.02)


def test_speaker_change_entirely_after_the_audio_boundary_falls_back():
    words = (SpeakerWord('One.', 0, 1, 1), SpeakerWord('Two.', 2.16, 2.4, 2))
    assert speaker_turns('One. Two.', words, 0, 2) == ()


def test_invalid_diarization_flag_fails_startup(monkeypatch):
    monkeypatch.setenv('NEMOTRON_DIARIZATION', 'tru')
    with pytest.raises(ValueError, match='true or false'):
        diarization_from_env()


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.reads = []
        self.listings = 0

    def get_paginator(self, name):
        assert name == 'list_objects_v2'
        return self

    def paginate(self, *, Bucket, Prefix, Delimiter):
        self.listings += 1
        folders = {Prefix + key[len(Prefix):].split('/')[0] + '/'
                   for key in self.objects if key.startswith(Prefix)}
        yield {'CommonPrefixes': [{'Prefix': folder} for folder in sorted(folders)]}

    def get_object(self, *, Bucket, Key):
        self.reads.append((Bucket, Key))
        data = self.objects[Key]
        return {'ContentLength': len(data), 'Body': io.BytesIO(data)}

    def enroll(self, tenant, speaker, label, number, *, sample_url=None):
        root = f'enrollment/{tenant}/speakers/{speaker}'
        self.objects[root + '/samples/one.wav'] = str(number).encode()
        self.objects[root + '/speaker.json'] = json.dumps({
            'speaker_id': speaker, 'label': label,
            'samples': [{'url': sample_url or f's3://enrolled/{root}/samples/one.wav'}],
        }).encode()


def gallery(client, monkeypatch, **kwargs):
    monkeypatch.setattr(EnrolledSpeakers, '_decode_sample', staticmethod(lambda data: np.array([int(data)])))
    return EnrolledSpeakers(client, 'enrolled', 'enrollment',
                           lambda audio: np.eye(8)[int(audio[0])], **kwargs)


def test_sixth_enrolled_person_matches_and_tenants_never_share_gallery(monkeypatch):
    client = MemoryS3()
    for i in range(6):
        client.enroll('tenant-a', f'id-{i}', f'Person {i}', i)
    client.enroll('tenant-b', 'id-5', 'Different tenant name', 5)
    mapper = gallery(client, monkeypatch)
    audio = np.full(24000, 5)
    assert mapper.identify('tenant-a', audio) == 'Person 5'
    assert mapper.identify('tenant-b', audio) == 'Different tenant name'
    assert mapper.identify('tenant-a', audio) == 'Person 5'
    assert client.listings == 2
    assert mapper.identify('../tenant-a', audio) == ''
    assert client.listings == 2


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'https://example.com/audio.wav',
                               's3://other/enrollment/tenant-a/speakers/id/samples/one.wav',
                               's3://enrolled/enrollment/tenant-b/speakers/id/samples/one.wav',
                               's3://enrolled/enrollment/tenant-a/speakers/id/samples/../one.wav'])
def test_manifest_cannot_read_outside_its_own_speaker(monkeypatch, url):
    client = MemoryS3()
    client.enroll('tenant-a', 'id', 'Name', 1, sample_url=url)
    mapper = gallery(client, monkeypatch)
    assert mapper.identify('tenant-a', np.full(24000, 1)) == ''
    assert len(client.reads) == 1  # Only the manifest; no sample URL was followed.


def test_ambiguous_or_low_similarity_stays_anonymous(monkeypatch):
    client = MemoryS3()
    client.enroll('tenant-a', 'one', 'One', 1)
    client.enroll('tenant-a', 'two', 'Two', 1)
    mapper = gallery(client, monkeypatch)
    assert mapper.identify('tenant-a', np.full(24000, 1)) == ''
    assert mapper.identify('tenant-a', np.full(24000, 7)) == ''


def test_gallery_expiry_and_lru_do_not_reuse_another_tenant(monkeypatch):
    client = MemoryS3()
    client.enroll('a', 'one', 'Before', 1)
    client.enroll('b', 'one', 'Other', 1)
    mapper = gallery(client, monkeypatch, ttl_s=0, max_tenants=1)
    audio = np.full(24000, 1)
    assert mapper.identify('a', audio) == 'Before'
    client.enroll('a', 'one', 'After', 1)
    assert mapper.identify('a', audio) == 'After'
    assert mapper.identify('b', audio) == 'Other'
    assert mapper.identify('a', audio) == 'After'
    assert len(mapper._cache) == 1


def test_storage_failure_is_anonymous_and_logs_no_object_details(monkeypatch, caplog):
    client = MemoryS3()
    def fail(*args, **kwargs):
        raise RuntimeError('private object path and credential')
    client.get_paginator = fail
    mapper = gallery(client, monkeypatch)
    assert mapper.identify('a', np.ones(24000)) == ''
    assert 'RuntimeError' in caplog.text
    assert 'private object' not in caplog.text


def test_oversized_gallery_is_not_silently_truncated(monkeypatch):
    client = MemoryS3()
    for i in range(257):
        client.enroll('a', f'id-{i:03}', f'Person {i}', 1)
    mapper = gallery(client, monkeypatch)
    assert mapper.identify('a', np.ones(24000)) == ''
    assert client.reads == []


def test_large_manifest_body_is_closed_and_not_decoded(monkeypatch):
    client = MemoryS3()
    client.enroll('a', 'one', 'One', 1)
    body = io.BytesIO(b'x' * 65537)
    client.get_object = lambda **kwargs: {'Body': body, 'ContentLength': 65537}
    mapper = gallery(client, monkeypatch)
    assert mapper.identify('a', np.ones(24000)) == ''
    assert body.closed


def test_cancelled_lookup_does_not_load_gallery_and_releases_lock(monkeypatch):
    client = MemoryS3()
    client.enroll('a', 'one', 'One', 1)
    mapper = gallery(client, monkeypatch)
    assert mapper.identify('a', np.ones(24000), is_active=lambda: False) == ''
    assert client.listings == 0
    assert mapper.identify('a', np.ones(24000)) == 'One'
