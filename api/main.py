"""FastAPI application entry point.

Run with:  uvicorn api.main:app --reload
Swagger UI: http://localhost:8000/docs
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any module reads configuration.
load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import llm as llm_pkg  # noqa: E402
import stt as stt_pkg  # noqa: E402
import tts_models as tts_pkg  # noqa: E402

from . import ari_routes, llm_routes, stt_routes, tts_routes, voice_routes  # noqa: E402
from .schemas import APIError, HealthResponse  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("voice-app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally preload models at startup (PRELOAD_MODELS=whisper,mms)."""
    for name in filter(None, (n.strip() for n in os.getenv("PRELOAD_MODELS", "").split(","))):
        try:
            if name in stt_pkg.available_stt_models():
                stt_pkg.get_stt_model(name).load()
            elif name in tts_pkg.available_tts_models():
                tts_pkg.get_tts_model(name).load()
            elif name in llm_pkg.available_llm_models():
                llm_pkg.get_llm_model(name).load()
            else:
                logger.warning("PRELOAD_MODELS: unknown model '%s'", name)
        except Exception:
            # Don't block startup; the error will surface again on first request.
            logger.exception("Could not preload '%s'", name)
    yield

app = FastAPI(
    title="Voice App - Phase 1",
    description="Standalone Speech-to-Text and Text-to-Speech API (Malayalam) for the WhatsApp voice agent.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],  # browsers reject credentials with a wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ---- Error handling: every error returns {"error": ..., "detail": ...} ----
@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.error, "detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p not in ("body", "form"))
        messages.append(f"{field}: {err['msg']}" if field else err["msg"])
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request", "detail": "; ".join(messages)},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Something went wrong. Check the server logs."},
    )


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
def health() -> HealthResponse:
    loaded = [n for n in stt_pkg.available_stt_models() if n in stt_pkg._instances and stt_pkg._instances[n].is_loaded]
    loaded += [n for n in tts_pkg.available_tts_models() if n in tts_pkg._instances and tts_pkg._instances[n].is_loaded]
    loaded += [n for n in llm_pkg.available_llm_models() if n in llm_pkg._instances and llm_pkg._instances[n].is_ready]
    return HealthResponse(
        stt_models=stt_pkg.available_stt_models(),
        tts_models=tts_pkg.available_tts_models(),
        llm_models=llm_pkg.available_llm_models(),
        loaded_models=loaded,
    )


app.include_router(stt_routes.router)
app.include_router(tts_routes.router)
app.include_router(llm_routes.router)
app.include_router(voice_routes.router)
app.include_router(ari_routes.router)

# Serve the frontend at "/" (mounted last so API routes take priority).
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Serve generated voice audio at /outputs/<file>.
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
if OUTPUTS_DIR.is_dir():
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
