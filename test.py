from PIL import Image
import torch

from main import MODEL_PATH, get_eval_transform, load_trained_model
from pretrained_bombek1 import load_bombek1_detector, predict_with_bombek1, resolve_detector_device

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AI_THRESHOLD = 0.50
UNCERTAINTY_MARGIN = 0.10
INFERENCE_BACKEND = "bombek1"  # Options: "bombek1", "local"
BOMBEK1_MODEL_DIR = "backend/models/bombek1"

# =========================
# LOAD MODEL
# =========================
def load_model():
    if INFERENCE_BACKEND == "bombek1":
        bombek1_device = resolve_detector_device(DEVICE)
        detector = load_bombek1_detector(
            device=bombek1_device,
            model_dir=BOMBEK1_MODEL_DIR,
        )
        return detector

    if INFERENCE_BACKEND == "local":
        model, _ = load_trained_model(model_path=MODEL_PATH, device=DEVICE)
        return model

    raise ValueError(f"Unsupported INFERENCE_BACKEND: {INFERENCE_BACKEND}")


model = load_model()
transform = get_eval_transform() if INFERENCE_BACKEND == "local" else None

# =========================
# PREDICT FUNCTION
# =========================
def normalize_image_path(raw_path):
    path = raw_path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in ("\"", "'"):
        path = path[1:-1]
    return path


def predict(image_path):
    if INFERENCE_BACKEND == "bombek1":
        _, confidence, ai_prob = predict_with_bombek1(model, image_path)
    else:
        image = Image.open(image_path).convert("RGB")
        variants = [
            image,
            image.transpose(Image.FLIP_LEFT_RIGHT),
        ]
        batch = torch.stack([transform(item) for item in variants]).to(DEVICE)

        with torch.no_grad():
            output = model(batch)
            probs = torch.softmax(output, dim=1).mean(dim=0)

        real_prob = probs[0].item()
        ai_prob = probs[1].item()
        confidence = max(real_prob, ai_prob)

    if ai_prob >= AI_THRESHOLD + UNCERTAINTY_MARGIN:
        label = "AI Generated"
    elif ai_prob <= AI_THRESHOLD - UNCERTAINTY_MARGIN:
        label = "Real"
    else:
        label = "Uncertain"

    return label, confidence, ai_prob


# =========================
# TEST IMAGE
# =========================
if __name__ == "__main__":
    image_path = normalize_image_path(input("Enter image path: "))

    label, confidence, ai_prob = predict(image_path)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"AI Probability: {ai_prob:.4f}")
