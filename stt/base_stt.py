"""Base class and shared helpers for Speech-to-Text models.

Every STT model implements `BaseSTT`. Models are loaded lazily (on first use)
so the API starts fast and tests never need to download weights.
"""
import logging
import shutil
import subprocess
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Sample rate expected by Whisper and IndicConformer.
TARGET_SAMPLE_RATE = 16000

# Containers/codecs that librosa/soundfile often cannot read on their own, so
# they are decoded through system (or bundled) ffmpeg instead.
FFMPEG_EXTENSIONS = (".webm", ".opus", ".ogg", ".m4a", ".mp4", ".mp3", ".aac")


class STTError(Exception):
    """Base error for all STT problems."""


class ModelLoadError(STTError):
    """The model could not be loaded (missing weights, no network, bad token...)."""


class TranscriptionError(STTError):
    """The audio could not be decoded or transcribed."""


def load_audio(path: str, sample_rate: int = TARGET_SAMPLE_RATE):
    """Decode any supported audio file to a mono float32 numpy array.

    Formats like mp3/m4a/webm need `ffmpeg` installed on the system.
    """
    # librosa/soundfile often cannot read browser MediaRecorder WebM/Opus on
    # macOS, so decode those containers through system or bundled ffmpeg.
    if path.lower().endswith(FFMPEG_EXTENSIONS):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError) as exc:
                raise TranscriptionError(
                    "Cannot decode browser recording: install ffmpeg with `brew install ffmpeg` "
                    "or install the bundled fallback with `pip install imageio-ffmpeg`."
                ) from exc
        try:
            import numpy as np

            result = subprocess.run(
                [ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
                check=True,
                capture_output=True,
            )
            audio = np.frombuffer(result.stdout, dtype=np.float32)
        except Exception as exc:
            raise TranscriptionError(f"Could not decode audio with ffmpeg: {exc}") from exc
        if audio.size == 0:
            raise TranscriptionError("The audio file contains no samples.")
        return audio

    try:
        import librosa

        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not decode audio file (is it a valid audio file, and is ffmpeg installed?): {exc}"
        ) from exc
    if audio.size == 0:
        raise TranscriptionError("The audio file contains no samples.")
    return audio


class BaseSTT(ABC):
    """Interface every STT model must follow."""

    #: Short identifier used in the API (e.g. "whisper").
    name: str = "base"

    def __init__(self) -> None:
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def _load(self) -> None:
        """Load weights into memory. Imports of heavy libraries go here."""

    @abstractmethod
    def _transcribe(self, audio_path: str, language: str) -> str:
        """Run inference and return raw text."""

    def load(self) -> None:
        """Load the model once; safe to call from multiple threads."""
        with self._lock:
            if self._loaded:
                return
            logger.info("Loading STT model '%s'...", self.name)
            try:
                self._load()
            except Exception as exc:
                logger.exception("Failed to load STT model '%s'", self.name)
                raise ModelLoadError(f"Could not load STT model '{self.name}': {exc}") from exc
            self._loaded = True
            logger.info("STT model '%s' ready.", self.name)

    def transcribe(self, audio_path: str, language: str = "ml") -> str:
        """Transcribe an audio file and return the recognised text."""
        self.load()
        try:
            return self._transcribe(audio_path, language).strip()
        except STTError:
            raise
        except Exception as exc:
            logger.exception("Transcription failed in '%s'", self.name)
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
