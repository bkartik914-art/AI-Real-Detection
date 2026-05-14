from dataclasses import dataclass
from io import BytesIO
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError

from pretrained_bombek1 import load_bombek1_detector, parse_bombek1_result, resolve_detector_device


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be treated as an image."""


@dataclass(frozen=True)
class DetectionResult:
    filename: str | None
    label: str
    raw_label: str
    upstream_label: str
    confidence: float
    ai_probability: float
    real_probability: float


def clamp_probability(value):
    return max(0.0, min(1.0, float(value)))


def classify_ai_probability(ai_probability, threshold, margin):
    if ai_probability >= threshold + margin:
        return "AI Generated", "ai"
    if ai_probability <= threshold - margin:
        return "Real", "real"
    return "Uncertain", "uncertain"


def decode_image(image_bytes):
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise ImageValidationError("Uploaded file is not a valid image.") from exc
    except OSError as exc:
        raise ImageValidationError("Uploaded image could not be decoded.") from exc


class DetectorService:
    def __init__(self, settings):
        self.settings = settings
        self._detector = None
        self._device_name = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @property
    def is_loaded(self):
        return self._detector is not None

    @property
    def device_name(self):
        return self._device_name or str(self.settings.model_device)

    def load(self):
        if self._detector is not None:
            return self._detector

        with self._load_lock:
            if self._detector is not None:
                return self._detector

            device_name = resolve_detector_device(self.settings.model_device)
            self._detector = load_bombek1_detector(
                device=device_name,
                model_dir=self.settings.model_dir,
            )
            self._device_name = device_name
            return self._detector

    def predict_bytes(self, image_bytes, filename=None):
        if not image_bytes:
            raise ImageValidationError("Uploaded file is empty.")
        if len(image_bytes) > self.settings.max_upload_bytes:
            raise ImageValidationError(
                f"Uploaded file is too large. Max size is {self.settings.max_upload_mb} MB."
            )

        image = decode_image(image_bytes)
        detector = self.load()

        # The model is large and GPU-backed in most setups; serialize calls to keep
        # memory usage predictable for a single-process API server.
        with self._predict_lock:
            raw_result = detector.predict(image)

        upstream_label, upstream_confidence, ai_probability = parse_bombek1_result(raw_result)
        ai_probability = clamp_probability(ai_probability)
        confidence = clamp_probability(upstream_confidence)
        real_probability = 1.0 - ai_probability
        display_label, raw_label = classify_ai_probability(
            ai_probability=ai_probability,
            threshold=self.settings.ai_threshold,
            margin=self.settings.uncertainty_margin,
        )

        return DetectionResult(
            filename=filename,
            label=display_label,
            raw_label=raw_label,
            upstream_label=upstream_label,
            confidence=confidence,
            ai_probability=ai_probability,
            real_probability=real_probability,
        )
