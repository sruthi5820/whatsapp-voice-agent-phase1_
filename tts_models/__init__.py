"""TTS package: registry so new models (Indic Parler-TTS, VITS...) can be plugged in later."""
import importlib
import threading

from .base_tts import (
    BaseTTS,
    SynthesisError,
    SynthesisResult,
    TTSError,
    TTSModelLoadError,
    normalize_loudness,
    save_wav,
)

__all__ = [
    "BaseTTS",
    "SynthesisError",
    "SynthesisResult",
    "TTSError",
    "TTSModelLoadError",
    "UnknownTTSModelError",
    "available_tts_models",
    "get_tts_model",
    "normalize_loudness",
    "register_tts_model",
    "save_wav",
]


class UnknownTTSModelError(TTSError):
    """Requested model name is not registered."""


_BUILTIN: dict[str, object] = {
    "mms": "tts_models.mms_tts:MMSTTS",
    "indic-parler-tts": "tts_models.indic_parler_tts:IndicParlerTTS",
    "maya": "tts_models.maya_tts:MayaTTS",
}
_instances: dict[str, BaseTTS] = {}
_lock = threading.Lock()


def register_tts_model(name: str, model_cls: type[BaseTTS]) -> None:
    """Register (or replace) a TTS model class under `name`."""
    with _lock:
        _BUILTIN[name] = model_cls
        _instances.pop(name, None)


def available_tts_models() -> list[str]:
    return sorted(_BUILTIN)


def get_tts_model(name: str) -> BaseTTS:
    """Return a cached model instance (weights load on first synthesis)."""
    with _lock:
        if name not in _BUILTIN:
            raise UnknownTTSModelError(
                f"Unknown TTS model '{name}'. Available: {', '.join(available_tts_models())}"
            )
        if name not in _instances:
            target = _BUILTIN[name]
            if isinstance(target, str):
                module_name, class_name = target.split(":")
                target = getattr(importlib.import_module(module_name), class_name)
            _instances[name] = target()
        return _instances[name]
