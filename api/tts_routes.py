"""POST /tts - text to speech."""
import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from tts_models import (
    SynthesisError,
    TTSModelLoadError,
    UnknownTTSModelError,
    available_tts_models,
    get_tts_model,
    save_wav,
)

from .schemas import APIError, ErrorResponse, TTSRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Text-to-Speech"])


def _output_dir() -> Path:
    """Read at call time so tests / .env changes take effect."""
    path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.get("/tts/models", summary="List TTS models")
def list_tts_models() -> dict:
    return {"models": available_tts_models(), "default": os.getenv("DEFAULT_TTS_MODEL", "indic-parler-tts")}


@router.post(
    "/tts",
    summary="Generate speech from text",
    response_class=FileResponse,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "WAV audio"},
        400: {"model": ErrorResponse, "description": "Unknown model"},
        422: {"model": ErrorResponse, "description": "Empty or invalid text"},
        500: {"model": ErrorResponse, "description": "Speech generation failed"},
        503: {"model": ErrorResponse, "description": "Model could not be loaded"},
    },
)
def synthesize(request: TTSRequest) -> FileResponse:
    model_name = (request.model or os.getenv("DEFAULT_TTS_MODEL", "indic-parler-tts")).strip().lower()
    try:
        tts_model = get_tts_model(model_name)
    except UnknownTTSModelError as exc:
        raise APIError(400, "Unknown TTS model", str(exc)) from exc

    try:
        result = tts_model.synthesize(request.text)
    except TTSModelLoadError as exc:
        raise APIError(503, "TTS model unavailable", str(exc)) from exc
    except SynthesisError as exc:
        raise APIError(500, "Speech generation failed", str(exc)) from exc

    filename = f"tts_{uuid.uuid4().hex[:12]}.wav"
    path = _output_dir() / filename
    try:
        save_wav(result, str(path))
    except OSError as exc:
        raise APIError(500, "Could not save audio", str(exc)) from exc

    logger.info("TTS '%s' wrote %s (%d chars)", model_name, path, len(request.text))
    return FileResponse(path, media_type="audio/wav", filename=filename)
