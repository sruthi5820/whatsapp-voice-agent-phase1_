"""Base class and helpers for Text-to-Speech models."""
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_TARGET_RMS_DBFS = -16.0
DEFAULT_PEAK_DBFS = -1.0
DEFAULT_MAX_GAIN_DB = 18.0
SILENCE_FLOOR_DBFS = -50.0


class TTSError(Exception):
    """Base error for all TTS problems."""


class TTSModelLoadError(TTSError):
    """The model could not be loaded."""


class SynthesisError(TTSError):
    """Speech could not be generated from the given text."""


@dataclass
class SynthesisResult:
    """Raw audio produced by a TTS model."""

    waveform: "numpy.ndarray"  # noqa: F821 - mono float32 in [-1, 1]
    sampling_rate: int


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _to_dbfs(value: float) -> float:
    import numpy as np

    return float(20.0 * np.log10(max(value, 1e-9)))


def normalize_loudness(
    waveform: "numpy.ndarray",  # noqa: F821
    sample_rate: int,
    target_rms_dbfs: float | None = None,
    peak_dbfs: float | None = None,
    max_gain_db: float | None = None,
) -> "numpy.ndarray":  # noqa: F821
    """Bring TTS audio up to a consistent, comfortably loud level.

    Maya and other neural vocoders leave a lot of headroom (their raw output is
    often ~-25 dBFS RMS / ~-11 dBFS peak), which sounds noticeably quieter than
    the samples on the Maya Research website. This raises the active-speech RMS
    to ``target_rms_dbfs`` and softly limits peaks to ``peak_dbfs`` so the result
    is as loud and clear as the web demo without clipping.

    Silence is ignored when measuring the level (so long pauses do not inflate
    the gain), DC offset is removed, and the gain is capped by ``max_gain_db``
    to avoid amplifying noise in near-silent clips.
    """
    import numpy as np

    audio = np.asarray(waveform, dtype="float32").reshape(-1)
    if audio.size == 0:
        return audio

    target_rms = (
        _env_float("TTS_TARGET_RMS_DBFS", DEFAULT_TARGET_RMS_DBFS)
        if target_rms_dbfs is None
        else float(target_rms_dbfs)
    )
    ceiling_db = (
        _env_float("TTS_PEAK_DBFS", DEFAULT_PEAK_DBFS)
        if peak_dbfs is None
        else float(peak_dbfs)
    )
    max_gain = (
        _env_float("TTS_MAX_GAIN_DB", DEFAULT_MAX_GAIN_DB)
        if max_gain_db is None
        else float(max_gain_db)
    )

    # Remove any DC offset before measuring.
    audio = audio - float(np.mean(audio, dtype="float64"))

    # Measure RMS over short frames, ignoring silence.
    frame = max(1, int(0.02 * max(1, sample_rate)))
    frames = audio.size // frame
    if frames:
        blocks = audio[: frames * frame].reshape(frames, frame).astype("float64")
        frame_rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        floor = 10.0 ** (SILENCE_FLOOR_DBFS / 20.0)
        active = frame_rms[frame_rms > floor]
        level = float(np.sqrt(np.mean((active if active.size else frame_rms) ** 2)))
    else:
        level = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))

    if level <= 1e-9:
        return audio

    gain_db = min(target_rms - _to_dbfs(level), max_gain)
    if gain_db > 0.0:
        audio = audio * np.float32(10.0 ** (gain_db / 20.0))

    # Soft limit the peaks so boosted audio never clips harshly.
    ceiling = float(10.0 ** (ceiling_db / 20.0))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        knee = 0.8 * ceiling
        span = max(ceiling - knee, 1e-6)
        np.clip(audio, -1.0, 1.0, out=audio)
        over = audio > knee
        if np.any(over):
            audio[over] = knee + span * np.tanh((audio[over] - knee) / span)
        under = audio < -knee
        if np.any(under):
            audio[under] = -knee - span * np.tanh((-audio[under] - knee) / span)

    return np.clip(audio, -1.0, 1.0).astype("float32")


def save_wav(result: SynthesisResult, path: str) -> None:
    """Write a SynthesisResult to a 16-bit PCM WAV file."""
    import numpy as np
    from scipy.io import wavfile

    samples = np.clip(result.waveform, -1.0, 1.0)
    wavfile.write(path, result.sampling_rate, (samples * 32767).astype(np.int16))


class BaseTTS(ABC):
    """Interface every TTS model must follow."""

    name: str = "base"

    def __init__(self) -> None:
        self._loaded = False
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def _load(self) -> None:
        """Load weights into memory."""

    @abstractmethod
    def _synthesize(self, text: str) -> SynthesisResult:
        """Run inference."""

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            logger.info("Loading TTS model '%s'...", self.name)
            try:
                self._load()
            except Exception as exc:
                logger.exception("Failed to load TTS model '%s'", self.name)
                raise TTSModelLoadError(f"Could not load TTS model '{self.name}': {exc}") from exc
            self._loaded = True
            logger.info("TTS model '%s' ready.", self.name)

    def synthesize(self, text: str) -> SynthesisResult:
        """Convert text to speech."""
        if not text or not text.strip():
            raise SynthesisError("Text is empty.")
        self.load()
        try:
            # Serialise inference: a single CPU/GPU model cannot run several
            # generations in parallel without thrashing and slowing everything.
            with self._infer_lock:
                result = self._synthesize(text.strip())
        except TTSError:
            raise
        except Exception as exc:
            logger.exception("Synthesis failed in '%s'", self.name)
            raise SynthesisError(f"Speech generation failed: {exc}") from exc

        if _env_enabled("TTS_NORMALIZE", True):
            try:
                result.waveform = normalize_loudness(result.waveform, result.sampling_rate)
            except Exception:
                logger.warning("Audio normalisation failed for '%s'", self.name, exc_info=True)
        return result
