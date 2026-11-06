"""STT package: a small registry so models can be added without touching the API.

To add a model: create a `BaseSTT` subclass, then either add it to `_BUILTIN`
below or call `register_stt_model("name", MyClass)` at import time.
"""
import importlib
import threading

from .base_stt import (
    BaseSTT,
    ModelLoadError,
    STTError,
    TranscriptionError,
)

__all__ = [
    "BaseSTT",
    "ModelLoadError",
    "STTError",
    "TranscriptionError",
    "UnknownSTTModelError",
    "available_stt_models",
    "get_stt_model",
    "register_stt_model",
]


class UnknownSTTModelError(STTError):
    """Requested model name is not registered."""


# name -> "module:Class" (imported lazily) or a class registered at runtime.
_BUILTIN: dict[str, object] = {
    "qwen_audio": "stt.qwen_audio_stt:QwenAudioSTT",
    "whisper": "stt.whisper_stt:WhisperSTT",
    "indicconformer": "stt.indicconformer_stt:IndicConformerSTT",
}
_instances: dict[str, BaseSTT] = {}
_lock = threading.Lock()


def register_stt_model(name: str, model_cls: type[BaseSTT]) -> None:
    """Register (or replace) an STT model class under `name`."""
    with _lock:
        _BUILTIN[name] = model_cls
        _instances.pop(name, None)


def available_stt_models() -> list[str]:
    return sorted(_BUILTIN)


def get_stt_model(name: str) -> BaseSTT:
    """Return a cached model instance (weights load on first transcription)."""
    with _lock:
        if name not in _BUILTIN:
            raise UnknownSTTModelError(
                f"Unknown STT model '{name}'. Available: {', '.join(available_stt_models())}"
            )
        if name not in _instances:
            target = _BUILTIN[name]
            if isinstance(target, str):
                module_name, class_name = target.split(":")
                target = getattr(importlib.import_module(module_name), class_name)
            _instances[name] = target()
        return _instances[name]
