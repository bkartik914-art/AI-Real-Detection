from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .detector_service import DetectorService, ImageValidationError
from .schemas import HealthResponse, MessageResponse, ModelInfo, PredictionResponse, Thresholds


settings = get_settings()
detector_service = DetectorService(settings)


def build_model_info():
    return ModelInfo(
        provider=settings.model_provider,
        backend=settings.model_backend,
        model_dir=str(settings.model_dir) if settings.model_dir else None,
        device=detector_service.device_name,
        loaded=detector_service.is_loaded,
    )


def build_thresholds():
    return Thresholds(
        ai_threshold=settings.ai_threshold,
        uncertainty_margin=settings.uncertainty_margin,
    )


@asynccontextmanager
async def lifespan(app):
    if settings.load_model_on_startup:
        await run_in_threadpool(detector_service.load)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_model=MessageResponse)
async def root():
    return MessageResponse(
        message="AI vs Real Detection API",
        health_url="/health",
        predict_url="/api/predict",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
        model=build_model_info(),
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if (
        content_type
        and content_type != "application/octet-stream"
        and content_type not in settings.allowed_image_types
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    try:
        image_bytes = await file.read()
        result = await run_in_threadpool(
            detector_service.predict_bytes,
            image_bytes,
            file.filename,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not ready: {exc}",
        ) from exc
    finally:
        await file.close()

    return PredictionResponse(
        filename=result.filename,
        label=result.label,
        raw_label=result.raw_label,
        upstream_label=result.upstream_label,
        confidence=result.confidence,
        ai_probability=result.ai_probability,
        real_probability=result.real_probability,
        thresholds=build_thresholds(),
        model=build_model_info(),
    )
