"""Indic Parler-TTS (ai4bharat/indic-parler-tts) for Malayalam.

A multilingual Indic TTS that takes two inputs:
  * ``prompt``     — the text to speak
  * ``description`` — a natural-language description of the speaking style

Requires the ``parler-tts`` library::

    pip install git+https://github.com/huggingface/parler-tts.git

Set in ``.env``::

    INDIC_PARLER_TTS_MODEL_ID=ai4bharat/indic-parler-tts
    # Optional style description (Malayalam female speaker, clear/close recording).
    INDIC_PARLER_TTS_DESCRIPTION="Anjali speaks clearly in a close-sounding environment with excellent recording quality, neutral tone and moderate pace."
"""
import logging
import os

from .base_tts import BaseTTS, SynthesisError, SynthesisResult

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTION = (
    "Anjali speaks clearly in a close-sounding environment with excellent recording "
    "quality, neutral tone and moderate pace."
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _resolve_device(torch) -> str:
    """Use the GPU only when enough memory is actually free."""
    requested = os.getenv("INDIC_PARLER_TTS_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            logger.warning("INDIC_PARLER_TTS_DEVICE=%s but CUDA is unavailable", requested)
        return "cpu"
    if requested in ("cuda", "cuda:0"):
        return "cuda"

    min_free_gb = _env_float("INDIC_PARLER_TTS_MIN_GPU_FREE_GB", 5.0)
    try:
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
    except Exception as exc:
        logger.warning("Could not query GPU memory (%s); using CPU for Indic Parler-TTS", exc)
        return "cpu"

    if free_gb < min_free_gb:
        logger.warning(
            "GPU has only %.2f GiB free (Indic Parler-TTS needs >= %.2f GiB); using CPU. "
            "Free GPU memory to use the GPU again.",
            free_gb,
            min_free_gb,
        )
        return "cpu"
    return "cuda"


def _is_oom(exc: Exception, torch) -> bool:
    oom_class = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_class) or "out of memory" in str(exc).lower()


def _resolve_dtype(torch, device: str):
    """Half precision on GPU; float32 on CPU (CPU half precision is slow)."""
    if device == "cpu":
        return torch.float32
    requested = os.getenv("INDIC_PARLER_TTS_DTYPE", "auto").strip().lower()
    if requested in ("float32", "fp32"):
        return torch.float32
    if requested in ("float16", "fp16", "half"):
        return torch.float16
    if requested in ("bfloat16", "bf16"):
        return torch.bfloat16
    # bf16 avoids T5 text-encoder overflow; fall back to fp16 on older GPUs.
    is_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    return torch.bfloat16 if is_bf16() else torch.float16


def _max_new_tokens(model_config, max_audio_seconds: float) -> int | None:
    if max_audio_seconds <= 0:
        return None
    sampling_rate = int(getattr(model_config, "sampling_rate", 44100))
    hop_length = int(getattr(model_config.audio_encoder, "hop_length", 512))
    frames = int(max_audio_seconds * sampling_rate / hop_length)
    return max(1, frames)


class IndicParlerTTS(BaseTTS):
    name = "indic-parler-tts"

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        self.model_id = os.path.expanduser(
            model_id or os.getenv("INDIC_PARLER_TTS_MODEL_ID", "ai4bharat/indic-parler-tts")
        )
        self.description = os.getenv("INDIC_PARLER_TTS_DESCRIPTION", DEFAULT_DESCRIPTION)
        self.max_audio_seconds = _env_float("INDIC_PARLER_TTS_MAX_AUDIO_SECONDS", 10.0)
        self._model = None
        self._tokenizer = None
        self._description_tokenizer = None
        self._device = "cpu"
        self._dtype = None

    def _load(self) -> None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        device = _resolve_device(torch)
        dtype = _resolve_dtype(torch, device)
        logger.info(
            "Indic Parler-TTS '%s' on %s (%s, max %.1fs audio)",
            self.model_id,
            device,
            dtype,
            self.max_audio_seconds,
        )

        try:
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id, low_cpu_mem_usage=True, torch_dtype=dtype
            )
        except Exception as exc:
            if dtype == torch.float32:
                raise
            logger.warning(
                "Could not load Indic Parler-TTS in %s (%s); retrying float32.", dtype, exc
            )
            dtype = torch.float32
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id, low_cpu_mem_usage=True, torch_dtype=dtype
            )

        if device == "cuda":
            try:
                model.to("cuda")
            except RuntimeError as exc:
                if not _is_oom(exc, torch):
                    raise
                logger.warning(
                    "CUDA out of memory loading Indic Parler-TTS; falling back to CPU."
                )
                torch.cuda.empty_cache()
                model.to(device="cpu", dtype=torch.float32)
                device, dtype = "cpu", torch.float32
        model.eval()

        self._model = model
        self._device = device
        self._dtype = dtype
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        enc_name = getattr(self._model.config.text_encoder, "_name_or_path", self.model_id)
        self._description_tokenizer = AutoTokenizer.from_pretrained(enc_name)
        logger.info(
            "Indic Parler-TTS '%s' ready on %s (%s)", self.model_id, self._device, self._dtype
        )

    def _generate(self, torch, text: str):
        description_input_ids = self._description_tokenizer(
            self.description, return_tensors="pt"
        ).to(self._device)
        prompt_input_ids = self._tokenizer(text, return_tensors="pt").to(self._device)
        kwargs = {
            "input_ids": description_input_ids.input_ids,
            "attention_mask": description_input_ids.attention_mask,
            "prompt_input_ids": prompt_input_ids.input_ids,
            "prompt_attention_mask": prompt_input_ids.attention_mask,
        }
        max_new_tokens = _max_new_tokens(self._model.config, self.max_audio_seconds)
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        with torch.inference_mode():
            return self._model.generate(**kwargs)

    def _synthesize(self, text: str) -> SynthesisResult:
        import torch

        if self._model is None:
            raise SynthesisError("Indic Parler-TTS model not loaded.")

        # If the model was loaded on CPU because the GPU was busy, upgrade it
        # automatically once enough GPU memory becomes available (no restart).
        if self._device == "cpu" and os.getenv("INDIC_PARLER_TTS_DEVICE", "auto").strip().lower() == "auto":
            try:
                free, _ = torch.cuda.mem_get_info()
                min_free = _env_float("INDIC_PARLER_TTS_MIN_GPU_FREE_GB", 5.0) * 1024**3
                if free >= min_free:
                    dtype = _resolve_dtype(torch, "cuda")
                    self._model.to(device="cuda", dtype=dtype)
                    self._device = "cuda"
                    self._dtype = dtype
                    logger.info(
                        "Moved Indic Parler-TTS to GPU (%s, %.2f GiB free)",
                        dtype,
                        free / 1024**3,
                    )
            except Exception:
                pass

        try:
            generation = self._generate(torch, text)
        except RuntimeError as exc:
            if self._device != "cuda" or not _is_oom(exc, torch):
                raise
            logger.warning(
                "CUDA out of memory during Indic Parler-TTS synthesis; moving to CPU."
            )
            torch.cuda.empty_cache()
            self._model.to(device="cpu", dtype=torch.float32)
            self._device = "cpu"
            self._dtype = torch.float32
            generation = self._generate(torch, text)

        audio = generation.float().cpu().numpy().reshape(-1).astype("float32")
        sampling_rate = int(getattr(self._model.config, "sampling_rate", 44100))
        if audio.size == 0:
            raise SynthesisError("Indic Parler-TTS produced no audio.")
        return SynthesisResult(waveform=audio, sampling_rate=sampling_rate)
