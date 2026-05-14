import os
from functools import lru_cache
from pathlib import Path

from pretrained_bombek1 import BOMBEK1_MODEL_DIR

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value, default):
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_optional_path(value):
    if value is None or not str(value).strip():
        return None
    return Path(value).resolve()


class Settings:
    app_name = os.getenv("APP_NAME", "AI vs Real Detection API")
    app_version = os.getenv("APP_VERSION", "1.0.0")

    model_backend = "bombek1"
    model_device = os.getenv("MODEL_DEVICE", "auto")
    model_provider = "local"
    model_dir = _parse_optional_path(os.getenv("BOMBEK1_MODEL_DIR")) or Path(BOMBEK1_MODEL_DIR).resolve()
    load_model_on_startup = _parse_bool(os.getenv("LOAD_MODEL_ON_STARTUP"), False)

    ai_threshold = float(os.getenv("AI_THRESHOLD", "0.50"))
    uncertainty_margin = float(os.getenv("UNCERTAINTY_MARGIN", "0.10"))
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "15"))

    cors_origins = _parse_csv(
        os.getenv("CORS_ORIGINS"),
        ["http://localhost:3000", "http://127.0.0.1:3000"],
    )
    allowed_image_types = set(
        _parse_csv(
            os.getenv("ALLOWED_IMAGE_TYPES"),
            [
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/webp",
                "image/bmp",
                "image/tiff",
            ],
        )
    )

    @property
    def max_upload_bytes(self):
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings():
    return Settings()
