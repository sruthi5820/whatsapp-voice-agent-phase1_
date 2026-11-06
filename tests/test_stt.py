"""STT tests. Models are faked so no weights are downloaded."""
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stt  # noqa: E402
from api.main import app  # noqa: E402
from stt.base_stt import BaseSTT, ModelLoadError, TranscriptionError  # noqa: E402


class FakeSTT(BaseSTT):
    name = "fake"

    def _load(self):
        pass

    def _transcribe(self, audio_path, language):
        return "  നമസ്കാരം  "


class BrokenLoadSTT(FakeSTT):
    def _load(self):
        raise RuntimeError("no weights")


class BrokenTranscribeSTT(FakeSTT):
    def _transcribe(self, audio_path, language):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def register_fakes():
    stt.register_stt_model("fake", FakeSTT)
    stt.register_stt_model("broken_load", BrokenLoadSTT)
    stt.register_stt_model("broken_run", BrokenTranscribeSTT)
    yield
    for name in ("fake", "broken_load", "broken_run"):
        stt._BUILTIN.pop(name, None)
        stt._instances.pop(name, None)


@pytest.fixture()
def client():
    return TestClient(app)


def _upload(client, name="clip.wav", data=b"RIFFxxxx", model="fake"):
    return client.post("/stt", files={"file": (name, io.BytesIO(data), "audio/wav")}, data={"model": model})


# ---- registry / base class ----
def test_registry_lists_builtin_models():
    models = stt.available_stt_models()
    assert "qwen_audio" in models and "whisper" in models and "indicconformer" in models


def test_unknown_model_raises():
    with pytest.raises(stt.UnknownSTTModelError):
        stt.get_stt_model("nope")


def test_transcribe_strips_text():
    assert stt.get_stt_model("fake").transcribe("x.wav") == "നമസ്കാരം"


def test_load_failure_is_wrapped():
    with pytest.raises(ModelLoadError):
        stt.get_stt_model("broken_load").transcribe("x.wav")


def test_transcription_failure_is_wrapped():
    with pytest.raises(TranscriptionError):
        stt.get_stt_model("broken_run").transcribe("x.wav")


# ---- API ----
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200 and res.json()["status"] == "ok"


def test_stt_success(client):
    res = _upload(client)
    assert res.status_code == 200
    body = res.json()
    assert body["text"] == "നമസ്കാരം" and body["model"] == "fake"


def test_stt_unsupported_format(client):
    res = _upload(client, name="notes.txt")
    assert res.status_code == 400 and res.json()["error"] == "Unsupported audio format"


def test_stt_unknown_model(client):
    res = _upload(client, model="does-not-exist")
    assert res.status_code == 400 and res.json()["error"] == "Unknown STT model"


def test_stt_empty_file(client):
    res = _upload(client, data=b"")
    assert res.status_code == 400 and res.json()["error"] == "Empty file"


def test_stt_missing_file(client):
    res = client.post("/stt", data={"model": "fake"})
    assert res.status_code == 422


def test_stt_file_too_large(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    res = _upload(client, data=b"x" * 10)
    assert res.status_code == 413


def test_stt_model_load_failure_returns_503(client):
    assert _upload(client, model="broken_load").status_code == 503


def test_stt_transcription_failure_returns_500(client):
    assert _upload(client, model="broken_run").status_code == 500
