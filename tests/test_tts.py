"""TTS tests. Models are faked so no weights are downloaded."""
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts_models as tts  # noqa: E402
from api.main import app  # noqa: E402
from tts_models.base_tts import BaseTTS, SynthesisError, SynthesisResult, save_wav  # noqa: E402


class FakeTTS(BaseTTS):
    name = "fake"

    def _load(self):
        pass

    def _synthesize(self, text):
        t = np.linspace(0, 0.2, 3200, endpoint=False)
        return SynthesisResult(waveform=(0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sampling_rate=16000)


class BrokenTTS(FakeTTS):
    def _synthesize(self, text):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def register_fakes(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    tts.register_tts_model("fake", FakeTTS)
    tts.register_tts_model("broken", BrokenTTS)
    yield
    for name in ("fake", "broken"):
        tts._BUILTIN.pop(name, None)
        tts._instances.pop(name, None)


@pytest.fixture()
def client():
    return TestClient(app)


# ---- registry / base class ----
def test_registry_lists_mms():
    assert "mms" in tts.available_tts_models()


def test_unknown_model_raises():
    with pytest.raises(tts.UnknownTTSModelError):
        tts.get_tts_model("nope")


def test_empty_text_rejected():
    with pytest.raises(SynthesisError):
        tts.get_tts_model("fake").synthesize("   ")


def test_unexpected_error_is_wrapped():
    with pytest.raises(SynthesisError):
        tts.get_tts_model("broken").synthesize("hello")


def test_save_wav_writes_riff_file(tmp_path):
    result = tts.get_tts_model("fake").synthesize("hello")
    path = tmp_path / "out.wav"
    save_wav(result, str(path))
    assert path.read_bytes()[:4] == b"RIFF"


def test_normalize_boosts_quiet_audio_without_clipping():
    from tts_models.base_tts import normalize_loudness

    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    quiet = (0.05 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    out = normalize_loudness(quiet, sr, target_rms_dbfs=-16, peak_dbfs=-1)

    before = float(np.sqrt(np.mean(quiet.astype("float64") ** 2)))
    after = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert after > before * 2
    assert float(np.max(np.abs(out))) <= 1.0


def test_normalize_leaves_silence_untouched():
    from tts_models.base_tts import normalize_loudness

    silence = np.zeros(2400, dtype="float32")
    out = normalize_loudness(silence, 24000)
    assert not np.any(out)
    assert out.dtype == np.float32


# ---- API ----
def test_tts_success_returns_wav(client, tmp_path):
    res = client.post("/tts", json={"text": "നമസ്കാരം", "model": "fake"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content[:4] == b"RIFF"
    assert len(list(tmp_path.glob("tts_*.wav"))) == 1


def test_tts_empty_text(client):
    res = client.post("/tts", json={"text": "   ", "model": "fake"})
    assert res.status_code == 422 and res.json()["error"] == "Invalid request"


def test_tts_missing_text(client):
    assert client.post("/tts", json={"model": "fake"}).status_code == 422


def test_tts_text_too_long(client, monkeypatch):
    monkeypatch.setenv("TTS_MAX_CHARS", "5")
    assert client.post("/tts", json={"text": "abcdefghij", "model": "fake"}).status_code == 422


def test_tts_unknown_model(client):
    res = client.post("/tts", json={"text": "hi", "model": "nope"})
    assert res.status_code == 400 and res.json()["error"] == "Unknown TTS model"


def test_tts_synthesis_failure_returns_500(client):
    assert client.post("/tts", json={"text": "hi", "model": "broken"}).status_code == 500
