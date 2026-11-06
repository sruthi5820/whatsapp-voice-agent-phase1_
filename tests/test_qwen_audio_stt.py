"""Qwen2-Audio configuration tests; no weights are loaded."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.qwen_audio_stt import QwenAudioSTT, _transcription_prompt  # noqa: E402


def test_local_model_directory_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_AUDIO_MODEL_ID", str(tmp_path))
    model = QwenAudioSTT()
    assert model.model_source == str(tmp_path)
    assert model.invalid_model_path == ""


def test_missing_absolute_model_directory_is_reported():
    model = QwenAudioSTT("/missing/qwen-audio-model")
    assert model.invalid_model_path == "/missing/qwen-audio-model"


def test_transcription_prompt_uses_language_name():
    assert _transcription_prompt("ml") == (
        "Transcribe the audio in Malayalam. Return only the spoken words."
    )