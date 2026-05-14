import argparse
import csv
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from main import MODEL_PATH, get_eval_transform, load_trained_model
from pretrained_bombek1 import load_bombek1_detector, predict_with_bombek1, resolve_detector_device


AI_THRESHOLD = 0.50
UNCERTAINTY_MARGIN = 0.10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CSV_FIELDS = [
    "image_path",
    "true_label",
    "predicted_label",
    "confidence",
    "ai_probability",
    "real_probability",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fast batch test best_model.pth on external images.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("external_eval"),
        help="Root folder containing images to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external_eval_predictions.csv"),
        help="CSV file to write predictions to.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device selection.",
    )
    parser.add_argument(
        "--backend",
        choices=["bombek1", "local"],
        default="bombek1",
        help="Inference backend. 'bombek1' uses the local pretrained detector.",
    )
    parser.add_argument(
        "--assume-label",
        choices=["ai", "real"],
        default=None,
        help="Treat every discovered image as this true label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to evaluate.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable flip test-time augmentation for faster inference.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    model, _ = load_trained_model(model_path=MODEL_PATH, device=device)
    return model


def load_backend_model(backend, device):
    if backend == "bombek1":
        bombek1_device = resolve_detector_device(device)
        return load_bombek1_detector(device=bombek1_device)

    return load_model(device)


def collect_image_paths(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_true_label(path):
    for part in reversed(path.parts):
        token = part.lower()
        if token == "ai":
            return "ai"
        if token in {"real", "nature"}:
            return "real"
    return None


class ExternalEvalDataset(Dataset):
    def __init__(self, image_paths, transform, assume_label=None):
        self.image_paths = image_paths
        self.transform = transform
        self.assume_label = assume_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return {
            "image_path": str(path),
            "true_label": self.assume_label or infer_true_label(path),
            "image": tensor,
        }


def build_loader(image_paths, transform, assume_label, batch_size, num_workers, device):
    dataset = ExternalEvalDataset(
        image_paths=image_paths,
        transform=transform,
        assume_label=assume_label,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def can_use_worker_processes(num_workers):
    if num_workers == 0:
        return True

    try:
        context = mp.get_context("spawn")
        queue = context.Queue()
        queue.close()
        queue.join_thread()
        return True
    except PermissionError:
        return False


def classify_ai_probability(score):
    if score >= AI_THRESHOLD + UNCERTAINTY_MARGIN:
        return "ai"
    if score <= AI_THRESHOLD - UNCERTAINTY_MARGIN:
        return "real"
    return "uncertain"


def classify_probabilities(probabilities):
    real_prob = probabilities[:, 0]
    ai_prob = probabilities[:, 1]
    confidence = torch.maximum(real_prob, ai_prob)

    predicted_labels = [classify_ai_probability(score) for score in ai_prob.tolist()]
    return predicted_labels, confidence.tolist(), ai_prob.tolist(), real_prob.tolist()


def update_summary(summary, true_labels, predicted_labels):
    summary["predicted_counts"].update(predicted_labels)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label:
            summary["labeled_count"] += 1
            summary["confusion"][(true_label, predicted_label)] += 1


def print_summary(summary, total_rows):
    predicted_counts = summary["predicted_counts"]
    print(f"Images evaluated: {total_rows}")
    print(f"Predicted AI: {predicted_counts.get('ai', 0)}")
    print(f"Predicted Real: {predicted_counts.get('real', 0)}")
    print(f"Predicted Uncertain: {predicted_counts.get('uncertain', 0)}")

    labeled_count = summary["labeled_count"]
    if labeled_count == 0:
        print("No true labels were available, so accuracy metrics were skipped.")
        return

    confusion = summary["confusion"]
    correct = confusion.get(("ai", "ai"), 0) + confusion.get(("real", "real"), 0)
    uncertain_count = confusion.get(("ai", "uncertain"), 0) + confusion.get(("real", "uncertain"), 0)
    accuracy = 100.0 * correct / labeled_count
    uncertain_rate = 100.0 * uncertain_count / labeled_count

    print(f"Labeled images: {labeled_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Uncertain rate: {uncertain_rate:.2f}%")
    print(
        "True AI -> "
        f"AI: {confusion.get(('ai', 'ai'), 0)}, "
        f"Real: {confusion.get(('ai', 'real'), 0)}, "
        f"Uncertain: {confusion.get(('ai', 'uncertain'), 0)}"
    )
    print(
        "True Real -> "
        f"AI: {confusion.get(('real', 'ai'), 0)}, "
        f"Real: {confusion.get(('real', 'real'), 0)}, "
        f"Uncertain: {confusion.get(('real', 'uncertain'), 0)}"
    )


def run_inference(model, loader, device, output_path, use_tta):
    total_rows = 0
    summary = {
        "predicted_counts": Counter(),
        "confusion": Counter(),
        "labeled_count": 0,
    }
    start_time = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    probabilities = torch.softmax(model(images), dim=1)
                    if use_tta:
                        flipped_images = torch.flip(images, dims=[3])
                        flipped_probabilities = torch.softmax(model(flipped_images), dim=1)
                        probabilities = (probabilities + flipped_probabilities) / 2.0

                predicted_labels, confidence, ai_probabilities, real_probabilities = classify_probabilities(probabilities)
                true_labels = list(batch["true_label"])
                image_paths = list(batch["image_path"])

                batch_rows = []
                for image_path, true_label, predicted_label, conf, ai_prob, real_prob in zip(
                    image_paths,
                    true_labels,
                    predicted_labels,
                    confidence,
                    ai_probabilities,
                    real_probabilities,
                ):
                    batch_rows.append(
                        {
                            "image_path": image_path,
                            "true_label": true_label or "",
                            "predicted_label": predicted_label,
                            "confidence": conf,
                            "ai_probability": ai_prob,
                            "real_probability": real_prob,
                        }
                    )

                writer.writerows(batch_rows)
                total_rows += len(batch_rows)
                update_summary(summary, true_labels, predicted_labels)

                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                print(f"Processed {total_rows}/{len(loader.dataset)} images ({rate:.1f} img/s)", flush=True)

    return total_rows, summary


def run_inference_bombek1(detector, image_paths, output_path, assume_label):
    total_rows = 0
    summary = {
        "predicted_counts": Counter(),
        "confusion": Counter(),
        "labeled_count": 0,
    }
    start_time = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for path in image_paths:
            true_label = assume_label or infer_true_label(path)
            _, confidence, ai_prob = predict_with_bombek1(detector, path)
            real_prob = 1.0 - ai_prob
            predicted_label = classify_ai_probability(ai_prob)

            writer.writerow(
                {
                    "image_path": str(path),
                    "true_label": true_label or "",
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "ai_probability": ai_prob,
                    "real_probability": real_prob,
                }
            )

            update_summary(summary, [true_label], [predicted_label])
            total_rows += 1

            elapsed = time.time() - start_time
            rate = total_rows / elapsed if elapsed > 0 else 0.0
            print(f"Processed {total_rows}/{len(image_paths)} images ({rate:.1f} img/s)", flush=True)

    return total_rows, summary


def main():
    args = parse_args()
    device = resolve_device(args.device)
    data_root = args.data.resolve()
    output_path = args.output.resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_root}")

    image_paths = collect_image_paths(data_root)
    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        raise FileNotFoundError(f"No image files found under: {data_root}")

    torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
    print(f"Images found: {len(image_paths)}")
    if args.backend == "local":
        print(f"Batch size: {args.batch_size}")
        print(f"TTA enabled: {not args.no_tta}")

    model = load_backend_model(args.backend, device)

    if args.backend == "bombek1":
        total_rows, summary = run_inference_bombek1(
            detector=model,
            image_paths=image_paths,
            output_path=output_path,
            assume_label=args.assume_label,
        )
    else:
        effective_num_workers = args.num_workers
        if not can_use_worker_processes(effective_num_workers):
            print(
                "Worker processes are blocked in this environment. "
                "Falling back to --num-workers 0.",
                flush=True,
            )
            effective_num_workers = 0
        print(f"Workers: {effective_num_workers}")

        transform = get_eval_transform()
        loader = build_loader(
            image_paths=image_paths,
            transform=transform,
            assume_label=args.assume_label,
            batch_size=args.batch_size,
            num_workers=effective_num_workers,
            device=device,
        )

        try:
            total_rows, summary = run_inference(
                model=model,
                loader=loader,
                device=device,
                output_path=output_path,
                use_tta=not args.no_tta,
            )
        except PermissionError:
            if effective_num_workers == 0:
                raise

            print(
                "Worker processes are blocked in this environment. "
                "Retrying with --num-workers 0.",
                flush=True,
            )
            loader = build_loader(
                image_paths=image_paths,
                transform=transform,
                assume_label=args.assume_label,
                batch_size=args.batch_size,
                num_workers=0,
                device=device,
            )
            total_rows, summary = run_inference(
                model=model,
                loader=loader,
                device=device,
                output_path=output_path,
                use_tta=not args.no_tta,
            )

    print(f"\nSaved predictions to: {output_path}")
    print_summary(summary, total_rows)


if __name__ == "__main__":
    main()
