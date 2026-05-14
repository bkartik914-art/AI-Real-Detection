import importlib.util
import sys
from pathlib import Path


BOMBEK1_MODEL_FILE = "model.py"
BOMBEK1_WEIGHTS_FILE = "pytorch_model.pt"
BOMBEK1_MODEL_DIR = Path(__file__).resolve().parent / "backend" / "models" / "bombek1"
_BOMBEK1_MODULE_NAME = "_bombek1_model"


def _missing_dependency_error(package_name):
    return RuntimeError(
        f"Missing dependency: {package_name}. "
        "Install with: py -3.11 -m pip install transformers timm peft pillow"
    )


def resolve_detector_device(device="auto"):
    try:
        import torch
    except Exception as exc:
        raise _missing_dependency_error("torch") from exc

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if hasattr(device, "type"):
        return device.type

    value = str(device).strip().lower()
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Bombek1, but CUDA is not available.")
    if value not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported Bombek1 device: {device}")
    return value


def _resolve_bombek1_files(model_dir):
    model_dir = Path(model_dir)
    model_file_path = model_dir / BOMBEK1_MODEL_FILE
    weights_file_path = model_dir / BOMBEK1_WEIGHTS_FILE

    missing_files = [
        str(path)
        for path in (model_file_path, weights_file_path)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Local Bombek1 model directory is missing required files: "
            + ", ".join(missing_files)
        )

    return model_file_path, weights_file_path


def _load_model_module(module_path):
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(_BOMBEK1_MODULE_NAME, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_BOMBEK1_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _ensure_numpy_pickle_compat():
    """
    Some checkpoints were serialized with NumPy 2.x and reference `numpy._core.*`.
    Older NumPy versions expose `numpy.core.*` instead. Add runtime aliases so
    torch.load can deserialize these checkpoints without forcing an immediate
    environment rebuild.
    """
    try:
        import numpy
    except Exception as exc:
        raise _missing_dependency_error("numpy") from exc

    try:
        import numpy._core  # noqa: F401
        return
    except Exception:
        pass

    import numpy.core as numpy_core

    sys.modules.setdefault("numpy._core", numpy_core)

    alias_pairs = [
        ("numpy.core.multiarray", "numpy._core.multiarray"),
        ("numpy.core.umath", "numpy._core.umath"),
        ("numpy.core.numeric", "numpy._core.numeric"),
        ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath"),
    ]

    for source_name, alias_name in alias_pairs:
        try:
            source_module = __import__(source_name, fromlist=["*"])
            sys.modules.setdefault(alias_name, source_module)
        except Exception:
            continue


def load_bombek1_detector(device="auto", model_dir=None, **_ignored_kwargs):
    device_name = resolve_detector_device(device)
    selected_model_dir = BOMBEK1_MODEL_DIR if model_dir is None else Path(model_dir)
    model_file_path, weights_file_path = _resolve_bombek1_files(selected_model_dir)
    _ensure_numpy_pickle_compat()
    module = _load_model_module(model_file_path)

    detector_cls = getattr(module, "AIImageDetector", None)
    if detector_cls is None:
        raise RuntimeError(
            "Bombek1 model.py did not expose AIImageDetector. "
            "The local model files may be incomplete or incompatible."
        )

    return detector_cls(model_path=str(weights_file_path), device=device_name)


def parse_bombek1_result(result):
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Bombek1 prediction result: {result}")

    ai_prob = float(result.get("probability", 0.0))
    confidence = float(result.get("confidence", max(ai_prob, 1.0 - ai_prob)))
    prediction = str(result.get("prediction", "")).strip().lower()

    if prediction in {"ai", "ai-generated", "generated", "fake"}:
        predicted_label = "ai"
    elif prediction in {"real", "human"}:
        predicted_label = "real"
    else:
        predicted_label = "ai" if ai_prob >= 0.5 else "real"

    return predicted_label, confidence, ai_prob


def predict_with_bombek1(detector, image_path):
    result = detector.predict(str(image_path))
    return parse_bombek1_result(result)
