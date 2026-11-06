"""POST /voice - full pipeline: STT → LLM → TTS."""
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from llm import CompletionError, LLMModelLoadError, UnknownLLMModelError, get_llm_model
from stt import (
    ModelLoadError as STTModelLoadError,
    TranscriptionError,
    UnknownSTTModelError,
    get_stt_model,
)
from tts_models import (
    SynthesisError,
    TTSModelLoadError,
    UnknownTTSModelError,
    get_tts_model,
    save_wav,
)

from .schemas import APIError, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Voice Assistant"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".flac", ".webm", ".aac"}
CHUNK_SIZE = 1024 * 1024

DEFAULT_SYSTEM = (
    "You are a helpful voice assistant. Reply in Malayalam with one short "
    "sentence of about 15-20 words. English technical words are fine; natural "
    "Malayalam-English mixing is allowed, and must sound natural for text-to-speech."
)


def _output_dir() -> Path:
    path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_reply(result) -> tuple[str, Path]:
    filename = f"voice_{uuid.uuid4().hex[:12]}.wav"
    path = _output_dir() / filename
    try:
        save_wav(result, str(path))
    except OSError as exc:
        raise APIError(500, "Could not save audio", str(exc)) from exc
    return filename, path


@router.post(
    "/voice/text",
    summary="Voice assistant (typed): LLM reply spoken back as Malayalam audio",
    response_class=FileResponse,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Reply audio (WAV)"},
        400: {"model": ErrorResponse, "description": "Empty prompt or unknown model"},
        500: {"model": ErrorResponse, "description": "Pipeline step failed"},
        503: {"model": ErrorResponse, "description": "A model could not be loaded/reached"},
    },
)
def voice_assistant_text(
    text: str = Form(..., description="Typed message from the caller"),
    llm_model: str = Form(None, description="LLM model name; defaults to DEFAULT_LLM_MODEL"),
    tts_model: str = Form(None, description="TTS model name; defaults to DEFAULT_TTS_MODEL"),
    system: str = Form(None, description="Optional system instruction for the LLM"),
) -> FileResponse:
    text = (text or "").strip()
    if not text:
        raise APIError(400, "Empty message", "Type a message before sending.")
    if len(text) > 4000:
        raise APIError(400, "Message too long", "Keep the typed message under 4000 characters.")

    llm_name = (llm_model or os.getenv("DEFAULT_LLM_MODEL", "remote")).strip().lower()
    tts_name = (tts_model or os.getenv("DEFAULT_TTS_MODEL", "indic-parler-tts")).strip().lower()

    try:
        llm = get_llm_model(llm_name)
    except UnknownLLMModelError as exc:
        raise APIError(400, "Unknown LLM model", str(exc)) from exc
    try:
        tts = get_tts_model(tts_name)
    except UnknownTTSModelError as exc:
        raise APIError(400, "Unknown TTS model", str(exc)) from exc

    start = time.perf_counter()
    try:
        reply = llm.complete(text, system or DEFAULT_SYSTEM)
        logger.info("Voice(text) LLM '%s': %s", llm_name, reply)
        result = tts.synthesize(reply)
        filename, path = _save_reply(result)
    except LLMModelLoadError as exc:
        raise APIError(503, "LLM unavailable", str(exc)) from exc
    except CompletionError as exc:
        raise APIError(500, "Completion failed", str(exc)) from exc
    except TTSModelLoadError as exc:
        raise APIError(503, "TTS model unavailable", str(exc)) from exc
    except SynthesisError as exc:
        raise APIError(500, "Speech generation failed", str(exc)) from exc

    elapsed = round(time.perf_counter() - start, 2)
    logger.info("Voice(text) pipeline finished in %.2fs", elapsed)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=filename,
        headers={
            "X-Input-Text": quote(text, safe=""),
            "X-Llm-Reply": quote(reply, safe=""),
            "X-Processing-Seconds": str(elapsed),
            "Access-Control-Expose-Headers": "X-Input-Text, X-Llm-Reply, X-Processing-Seconds",
        },
    )


@router.post(
    "/voice",
    summary="Voice assistant: transcribe, reply with Ollama, speak back",
    response_class=FileResponse,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Reply audio (WAV)"},
        400: {"model": ErrorResponse, "description": "Bad file, format, model, or empty prompt"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Pipeline step failed"},
        503: {"model": ErrorResponse, "description": "A model could not be loaded/reached"},
    },
)
def voice_assistant(
    file: UploadFile = File(..., description="Audio file with the user's spoken message"),
    stt_model: str = Form(None, description="STT model name; defaults to DEFAULT_STT_MODEL"),
    language: str = Form(None, description="ISO language code, e.g. 'ml'; defaults to DEFAULT_LANGUAGE"),
    llm_model: str = Form(None, description="LLM model name; defaults to DEFAULT_LLM_MODEL"),
    tts_model: str = Form(None, description="TTS model name; defaults to DEFAULT_TTS_MODEL"),
    system: str = Form(None, description="Optional system instruction for the LLM"),
) -> FileResponse:
    stt_name = (stt_model or os.getenv("DEFAULT_STT_MODEL", "whisper")).strip().lower()
    language = (language or os.getenv("DEFAULT_LANGUAGE", "ml")).strip().lower()
    llm_name = (llm_model or os.getenv("DEFAULT_LLM_MODEL", "remote")).strip().lower()
    tts_name = (tts_model or os.getenv("DEFAULT_TTS_MODEL", "indic-parler-tts")).strip().lower()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise APIError(
            400,
            "Unsupported audio format",
            f"Got '{suffix or 'no extension'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Resolve models up front (weights load lazily on first use).
    try:
        stt = get_stt_model(stt_name)
    except UnknownSTTModelError as exc:
        raise APIError(400, "Unknown STT model", str(exc)) from exc
    try:
        llm = get_llm_model(llm_name)
    except UnknownLLMModelError as exc:
        raise APIError(400, "Unknown LLM model", str(exc)) from exc
    try:
        tts = get_tts_model(tts_name)
    except UnknownTTSModelError as exc:
        raise APIError(400, "Unknown TTS model", str(exc)) from exc

    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
    tmp_path = None
    start = time.perf_counter()
    try:
        # --- STT ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := file.file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise APIError(413, "File too large", f"Maximum upload size is {max_bytes // (1024 * 1024)} MB.")
                tmp.write(chunk)
        if size == 0:
            raise APIError(400, "Empty file", "The uploaded file has no content.")

        text = stt.transcribe(tmp_path, language)
        logger.info("Voice STT '%s' [%s]: %s", stt_name, language, text or "(empty)")
        if not text.strip():
            raise APIError(
                400,
                "No speech detected",
                "The recording was silent or too short. Try again or type your message.",
            )

        # --- LLM ---
        system_prompt = system or DEFAULT_SYSTEM
        reply = llm.complete(text, system_prompt)
        logger.info("Voice LLM '%s': %s", llm_name, reply)

        # --- TTS ---
        result = tts.synthesize(reply)

        filename = f"voice_{uuid.uuid4().hex[:12]}.wav"
        path = _output_dir() / filename
        try:
            save_wav(result, str(path))
        except OSError as exc:
            raise APIError(500, "Could not save audio", str(exc)) from exc

        elapsed = round(time.perf_counter() - start, 2)
        logger.info("Voice pipeline finished in %.2fs", elapsed)

        # Return the WAV; the STT text and LLM reply ride along in headers so
        # the frontend can display them.
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=filename,
            headers={
                "X-Stt-Text": quote(text, safe=""),
                "X-Llm-Reply": quote(reply, safe=""),
                "X-Processing-Seconds": str(elapsed),
                "Access-Control-Expose-Headers": "X-Stt-Text, X-Llm-Reply, X-Processing-Seconds",
            },
        )
    except STTModelLoadError as exc:
        raise APIError(503, "STT model unavailable", str(exc)) from exc
    except TranscriptionError as exc:
        raise APIError(500, "Transcription failed", str(exc)) from exc
    except LLMModelLoadError as exc:
        raise APIError(503, "LLM unavailable", str(exc)) from exc
    except CompletionError as exc:
        raise APIError(500, "Completion failed", str(exc)) from exc
    except TTSModelLoadError as exc:
        raise APIError(503, "TTS model unavailable", str(exc)) from exc
    except SynthesisError as exc:
        raise APIError(500, "Speech generation failed", str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
