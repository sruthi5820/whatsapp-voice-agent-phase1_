"""Tests for the typed version of the voice pipeline (no STT / no real models)."""
import os
import sys

from urllib.parse import unquote

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import voice_routes  # noqa: E402
from api.main import app  # noqa: E402
from tts_models import SynthesisResult  # noqa: E402


class FakeLLM:
    def complete(self, text, system):
        return f"reply to {text}"


class FakeTTS:
    def synthesize(self, text):
        return SynthesisResult(waveform=np.zeros(800, dtype=np.float32), sampling_rate=16000)


def test_typed_message_returns_audio_and_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_routes, "get_llm_model", lambda name: FakeLLM())
    monkeypatch.setattr(voice_routes, "get_tts_model", lambda name: FakeTTS())
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    client = TestClient(app)
    response = client.post("/voice/text", data={"text": "ഹായ്"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert unquote(response.headers["X-Input-Text"]) == "ഹായ്"
    assert unquote(response.headers["X-Llm-Reply"]) == "reply to ഹായ്"
    assert list(tmp_path.glob("voice_*.wav"))


def test_typed_message_rejects_blank_text(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    client = TestClient(app)
    response = client.post("/voice/text", data={"text": "   "})
    assert response.status_code == 400
