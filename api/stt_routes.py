"""POST /stt - speech to text."""
import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from stt import (
    ModelLoadError,
    TranscriptionError,
    UnknownSTTModelError,
    available_stt_models,
    get_stt_model,
)

from .schemas import APIError, ErrorResponse, STTModelsResponse, STTResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Speech-to-Text"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".flac", ".webm", ".aac"}
CHUNK_SIZE = 1024 * 1024  # read uploads 1 MB at a time


def _default_model() -> str:
    return os.getenv("DEFAULT_STT_MODEL", "qwen_audio")


@router.get("/stt/models", response_model=STTModelsResponse, summary="List STT models")
def list_stt_models() -> STTModelsResponse:
    return STTModelsResponse(models=available_stt_models(), default=_default_model())


@router.post(
    "/stt",
    response_model=STTResponse,
    summary="Transcribe an audio file",
    responses={
        400: {"model": ErrorResponse, "description": "Bad file, format or model name"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Transcription failed"},
        503: {"model": ErrorResponse, "description": "Model could not be loaded"},
    },
)
def transcribe(
    file: UploadFile = File(..., description="Audio file (wav, mp3, ogg, opus, m4a, mp4, flac, webm, aac)"),
    model: str = Form(None, description="STT model name; defaults to DEFAULT_STT_MODEL"),
    language: str = Form(None, description="ISO language code, e.g. 'ml'; defaults to DEFAULT_LANGUAGE"),
) -> STTResponse:
    model_name = (model or _default_model()).strip().lower()
    language = (language or os.getenv("DEFAULT_LANGUAGE", "ml")).strip().lower()

    # 1. Validate the file extension before doing any work.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise APIError(
            400,
            "Unsupported audio format",
            f"Got '{suffix or 'no extension'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Resolve the model (cheap; weights load later on first use).
    try:
        stt_model = get_stt_model(model_name)
    except UnknownSTTModelError as exc:
        raise APIError(400, "Unknown STT model", str(exc)) from exc

    # 3. Save the upload to a temp file, enforcing the size limit while reading.
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
    tmp_path = None
    try:
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

        # 4. Transcribe.
        start = time.perf_counter()
        text = stt_model.transcribe(tmp_path, language)
        elapsed = round(time.perf_counter() - start, 2)
        logger.info(
            "STT '%s' [%s] finished in %.2fs: %s",
            model_name,
            language,
            elapsed,
            text or "(empty)",
        )
        return STTResponse(text=text, model=model_name, language=language, processing_seconds=elapsed)

    except ModelLoadError as exc:
        raise APIError(503, "STT model unavailable", str(exc)) from exc
    except TranscriptionError as exc:
        raise APIError(500, "Transcription failed", str(exc)) from exc
    finally:
        # Always remove the temp file, even on errors.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
