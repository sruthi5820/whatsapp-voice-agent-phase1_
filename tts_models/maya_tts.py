"""Maya 2 Native (maya-research/maya2-native) for Malayalam and other Indian languages.

Note: the engine ships its own top-level ``tts`` package, so this project's TTS
package is named ``tts_models`` to avoid shadowing it.

Reference engine: ``maya3.Maya3TTS`` (Apache-2.0). Generates 24 kHz mono speech and
supports streaming, voice selection and zero-shot cloning.

Install the engine (Linux + NVIDIA CUDA only)::

    pip install "maya3[server] @ git+https://github.com/MayaResearch/maya2-native" \
        --extra-index-url https://download.pytorch.org/whl/cu128

Set in ``.env``::

    MAYA_TTS_MODEL_ID=/path/to/models/maya2-native
    MAYA_TTS_VOICE=Ahana
    MAYA_TTS_LANG=ml
    # DEFAULT_TTS_MODEL=maya

Built-in voices that cover Malayalam (``ml``): Ahana, Vaidehi, Malini, Charvi,
Jahnavi, Anirban, Nakul, Vasant, Siddhant, Satvik.
"""
import logging
import os

from .base_tts import BaseTTS, SynthesisError, SynthesisResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "maya-research/maya2-native"
DEFAULT_VOICE = "Ahana"
DEFAULT_LANG = "ml"
SAMPLE_RATE = 24000


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


class MayaTTS(BaseTTS):
    name = "maya"

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        self.model_id = os.path.expanduser(
            model_id or _env_str("MAYA_TTS_MODEL_ID", DEFAULT_MODEL_ID)
        )
        self.voice = _env_str("MAYA_TTS_VOICE", DEFAULT_VOICE)
        self.lang = _env_str("MAYA_TTS_LANG", DEFAULT_LANG)
        self.normalize = os.getenv("MAYA_TTS_NORMALIZE", "1").lower() not in ("0", "false", "no")
        self.max_seats = int(os.getenv("MAYA_TTS_MAX_SEATS", "1"))
        self._engine = None

    def _load(self) -> None:
        try:
            from maya3 import Maya3TTS
        except ImportError as exc:
            raise SynthesisError(
                "Maya 2 Native requires the 'maya3' package. Install it with: "
                "pip install \"maya3[server] @ git+https://github.com/MayaResearch/maya2-native\" "
                "--extra-index-url https://download.pytorch.org/whl/cu128"
            ) from exc

        logger.info(
            "Maya 2 Native '%s' voice=%s lang=%s normalize=%s max_seats=%d",
            self.model_id,
            self.voice,
            self.lang,
            self.normalize,
            self.max_seats,
        )

        try:
            engine = Maya3TTS.from_pretrained(self.model_id, max_seats=self.max_seats)
        except Exception as exc:
            logger.exception("Failed to load Maya 2 Native from '%s'", self.model_id)
            raise SynthesisError(f"Could not load Maya 2 Native: {exc}") from exc

        self._engine = engine
        logger.info("Maya 2 Native '%s' ready (voice=%s, lang=%s)", self.model_id, self.voice, self.lang)

    def _synthesize(self, text: str) -> SynthesisResult:
        if self._engine is None:
            raise SynthesisError("Maya 2 Native engine not loaded.")

        try:
            waveform = self._engine.generate(
                text,
                lang=self.lang,
                voice=self.voice,
                normalize=self.normalize,
            )
        except Exception as exc:
            logger.exception("Maya 2 Native synthesis failed")
            raise SynthesisError(f"Maya 2 Native synthesis failed: {exc}") from exc

        import numpy as np

        audio = np.asarray(waveform, dtype="float32").reshape(-1)
        if audio.size == 0:
            raise SynthesisError("Maya 2 Native produced no audio.")
        return SynthesisResult(waveform=audio, sampling_rate=SAMPLE_RATE)

    def close(self) -> None:
        """Release the engine and any GPU memory it holds."""
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as exc:
                logger.warning("Error closing Maya 2 Native engine: %s", exc)
            self._engine = None
