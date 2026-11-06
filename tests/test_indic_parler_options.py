"""Indic Parler-TTS device selection tests. No torch or weights required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_models.indic_parler_tts import _resolve_device  # noqa: E402


class FakeCuda:
    def __init__(self, free_bytes=0, available=True, fail=False):
        self.free_bytes = free_bytes
        self.available = available
        self.fail = fail
        self.OutOfMemoryError = RuntimeError

    def is_available(self):
        return self.available

    def mem_get_info(self):
        if self.fail:
            raise RuntimeError("CUDA error: out of memory")
        return self.free_bytes, 16 * 1024**3


class FakeTorch:
    def __init__(self, **kwargs):
        self.cuda = FakeCuda(**kwargs)


def test_auto_uses_cpu_when_gpu_low(monkeypatch):
    monkeypatch.delenv("INDIC_PARLER_TTS_DEVICE", raising=False)
    monkeypatch.setenv("INDIC_PARLER_TTS_MIN_GPU_FREE_GB", "5")
    assert _resolve_device(FakeTorch(free_bytes=1 * 1024**3)) == "cpu"


def test_auto_uses_gpu_when_enough_memory(monkeypatch):
    monkeypatch.delenv("INDIC_PARLER_TTS_DEVICE", raising=False)
    monkeypatch.setenv("INDIC_PARLER_TTS_MIN_GPU_FREE_GB", "5")
    assert _resolve_device(FakeTorch(free_bytes=10 * 1024**3)) == "cuda"


def test_explicit_cpu_and_cuda(monkeypatch):
    monkeypatch.setenv("INDIC_PARLER_TTS_DEVICE", "cpu")
    assert _resolve_device(FakeTorch(free_bytes=10 * 1024**3)) == "cpu"
    monkeypatch.setenv("INDIC_PARLER_TTS_DEVICE", "cuda")
    assert _resolve_device(FakeTorch(free_bytes=0)) == "cuda"


def test_memory_query_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("INDIC_PARLER_TTS_DEVICE", raising=False)
    assert _resolve_device(FakeTorch(fail=True)) == "cpu"


class FakeCudaDtype:
    def __init__(self, bf16):
        self._bf16 = bf16

    def is_bf16_supported(self):
        return self._bf16


class FakeTorchDtype:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, bf16=False):
        self.cuda = FakeCudaDtype(bf16)


def test_dtype_is_float32_on_cpu(monkeypatch):
    from tts_models.indic_parler_tts import _resolve_dtype

    monkeypatch.delenv("INDIC_PARLER_TTS_DTYPE", raising=False)
    assert _resolve_dtype(FakeTorchDtype(), "cpu") == "float32"


def test_auto_dtype_prefers_bf16_then_fp16(monkeypatch):
    from tts_models.indic_parler_tts import _resolve_dtype

    monkeypatch.delenv("INDIC_PARLER_TTS_DTYPE", raising=False)
    assert _resolve_dtype(FakeTorchDtype(bf16=True), "cuda") == "bfloat16"
    assert _resolve_dtype(FakeTorchDtype(bf16=False), "cuda") == "float16"


def test_explicit_dtype_is_honoured(monkeypatch):
    from tts_models.indic_parler_tts import _resolve_dtype

    monkeypatch.setenv("INDIC_PARLER_TTS_DTYPE", "fp32")
    assert _resolve_dtype(FakeTorchDtype(bf16=True), "cuda") == "float32"


def test_max_new_tokens_from_audio_seconds():
    from types import SimpleNamespace
    from tts_models.indic_parler_tts import _max_new_tokens

    config = SimpleNamespace(sampling_rate=44100, audio_encoder=SimpleNamespace(hop_length=512))
    assert _max_new_tokens(config, 10) == 861
    assert _max_new_tokens(config, 0) is None
