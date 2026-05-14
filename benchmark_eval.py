import argparse
import csv
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from main import MODEL_PATH, get_eval_transform, load_trained_model
from pretrained_bombek1 import load_bombek1_detector, predict_with_bombek1, resolve_detector_device


AI_THRESHOLD = 0.50
UNCERTAINTY_MARGIN = 0.10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PREDICTION_FIELDS = [
    "image_path",
    "dataset_type",
    "source_name",
    "true_label",
    "predicted_label",
    "confidence",
    "ai_probability",
    "real_probability",
]
SUMMARY_FIELDS = [
    "source_name",
    "dataset_type",
    "true_label",
    "total_images",
    "predicted_ai",
    "predicted_real",
    "predicted_uncertain",
    "accuracy",
    "precision_ai",
    "recall_ai",
    "f1_ai",
    "ai_detect_rate",
    "ai_missed_as_real_rate",
    "ai_uncertain_rate",
    "real_accept_rate",
    "real_flagged_as_ai_rate",
    "real_uncertain_rate",
    "false_positive_rate",
    "false_negative_rate",
    "uncertain_rate",
    "avg_ai_probability",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark best_model.pth on external real and AI image folders."
    )
    parser.add_argument(
        "--real-data",
        type=Path,
        default=Path("external_eval"),
        help="Root folder of real images.",
    )
    parser.add_argument(
        "--ai-data",
        type=Path,
        default=Path("external_eval_ai"),
        help="Root folder of AI-generated images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_predictions.csv"),
        help="CSV file with per-image predictions.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark_summary.csv"),
        help="CSV file with overall and per-source metrics.",
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
        "--limit-real",
        type=int,
        default=None,
        help="Optional max number of real images to evaluate.",
    )
    parser.add_argument(
        "--limit-ai",
        type=int,
        default=None,
        help="Optional max number of AI images to evaluate.",
    )
    parser.add_argument(
        "--real-group-mode",
        choices=["root", "top-level"],
        default="root",
        help="How to group real-image sources in the summary.",
    )
    parser.add_argument(
        "--ai-group-mode",
        choices=["root", "top-level"],
        default="top-level",
        help="How to group AI-image sources in the summary.",
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


def get_source_name(path, root, dataset_type, group_mode):
    relative = path.relative_to(root)
    if group_mode == "root":
        return root.name
    if len(relative.parts) > 1:
        return relative.parts[0]
    return f"{dataset_type}_root"


def build_records(root, dataset_type, true_label, group_mode, limit=None):
    if not root.exists():
        print(f"Skipping missing folder: {root}")
        return []

    image_paths = collect_image_paths(root)
    if limit is not None:
        image_paths = image_paths[:limit]

    records = []
    for path in image_paths:
        records.append(
            {
                "image_path": path,
                "dataset_type": dataset_type,
                "source_name": get_source_name(path, root, dataset_type, group_mode),
                "true_label": true_label,
            }
        )
    return records


class BenchmarkDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return {
            "image_path": str(record["image_path"]),
            "dataset_type": record["dataset_type"],
            "source_name": record["source_name"],
            "true_label": record["true_label"],
            "image": tensor,
        }


def build_loader(records, transform, batch_size, num_workers, device):
    dataset = BenchmarkDataset(records=records, transform=transform)
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


def create_metrics_counter():
    return {
        "total": 0,
        "predicted_counts": Counter(),
        "confusion": Counter(),
        "ai_probability_sum": 0.0,
    }


def update_metrics(metrics, true_label, predicted_label, ai_probability):
    metrics["total"] += 1
    metrics["predicted_counts"][predicted_label] += 1
    metrics["confusion"][(true_label, predicted_label)] += 1
    metrics["ai_probability_sum"] += ai_probability


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metric_row(source_name, dataset_type, true_label, metrics):
    total = metrics["total"]
    predicted_counts = metrics["predicted_counts"]
    confusion = metrics["confusion"]

    tp = confusion.get(("ai", "ai"), 0)
    tn = confusion.get(("real", "real"), 0)
    fp = confusion.get(("real", "ai"), 0)
    ai_real = confusion.get(("ai", "real"), 0)
    ai_uncertain = confusion.get(("ai", "uncertain"), 0)
    real_ai = confusion.get(("real", "ai"), 0)
    real_uncertain = confusion.get(("real", "uncertain"), 0)
    fn = ai_real + ai_uncertain
    real_total = confusion.get(("real", "ai"), 0) + confusion.get(("real", "real"), 0) + confusion.get(("real", "uncertain"), 0)
    ai_total = confusion.get(("ai", "ai"), 0) + confusion.get(("ai", "real"), 0) + confusion.get(("ai", "uncertain"), 0)
    accuracy = safe_divide(tp + tn, total)
    precision_ai = safe_divide(tp, tp + fp)
    recall_ai = safe_divide(tp, ai_total)
    f1_ai = safe_divide(2 * precision_ai * recall_ai, precision_ai + recall_ai) if (precision_ai + recall_ai) else 0.0
    ai_detect_rate = safe_divide(tp, ai_total)
    ai_missed_as_real_rate = safe_divide(ai_real, ai_total)
    ai_uncertain_rate = safe_divide(ai_uncertain, ai_total)
    real_accept_rate = safe_divide(tn, real_total)
    real_flagged_as_ai_rate = safe_divide(real_ai, real_total)
    real_uncertain_rate = safe_divide(real_uncertain, real_total)
    false_positive_rate = safe_divide(fp, real_total)
    false_negative_rate = safe_divide(fn, ai_total)
    uncertain_rate = safe_divide(predicted_counts.get("uncertain", 0), total)
    avg_ai_probability = safe_divide(metrics["ai_probability_sum"], total)

    return {
        "source_name": source_name,
        "dataset_type": dataset_type,
        "true_label": true_label,
        "total_images": total,
        "predicted_ai": predicted_counts.get("ai", 0),
        "predicted_real": predicted_counts.get("real", 0),
        "predicted_uncertain": predicted_counts.get("uncertain", 0),
        "accuracy": accuracy,
        "precision_ai": precision_ai,
        "recall_ai": recall_ai,
        "f1_ai": f1_ai,
        "ai_detect_rate": ai_detect_rate,
        "ai_missed_as_real_rate": ai_missed_as_real_rate,
        "ai_uncertain_rate": ai_uncertain_rate,
        "real_accept_rate": real_accept_rate,
        "real_flagged_as_ai_rate": real_flagged_as_ai_rate,
        "real_uncertain_rate": real_uncertain_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "uncertain_rate": uncertain_rate,
        "avg_ai_probability": avg_ai_probability,
    }


def write_summary_csv(summary_rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def run_inference(model, loader, device, output_path, use_tta):
    overall_metrics = create_metrics_counter()
    source_metrics = defaultdict(create_metrics_counter)
    total_rows = 0
    start_time = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_FIELDS)
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
                image_paths = list(batch["image_path"])
                dataset_types = list(batch["dataset_type"])
                source_names = list(batch["source_name"])
                true_labels = list(batch["true_label"])

                batch_rows = []
                for image_path, dataset_type, source_name, true_label, predicted_label, conf, ai_prob, real_prob in zip(
                    image_paths,
                    dataset_types,
                    source_names,
                    true_labels,
                    predicted_labels,
                    confidence,
                    ai_probabilities,
                    real_probabilities,
                ):
                    batch_rows.append(
                        {
                            "image_path": image_path,
                            "dataset_type": dataset_type,
                            "source_name": source_name,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "confidence": conf,
                            "ai_probability": ai_prob,
                            "real_probability": real_prob,
                        }
                    )
                    update_metrics(overall_metrics, true_label, predicted_label, ai_prob)
                    update_metrics(source_metrics[(source_name, dataset_type, true_label)], true_label, predicted_label, ai_prob)

                writer.writerows(batch_rows)
                total_rows += len(batch_rows)

                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                print(f"Processed {total_rows}/{len(loader.dataset)} images ({rate:.1f} img/s)", flush=True)

    return total_rows, overall_metrics, source_metrics


def run_inference_bombek1(detector, records, output_path):
    overall_metrics = create_metrics_counter()
    source_metrics = defaultdict(create_metrics_counter)
    total_rows = 0
    start_time = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()

        for record in records:
            image_path = record["image_path"]
            dataset_type = record["dataset_type"]
            source_name = record["source_name"]
            true_label = record["true_label"]

            _, confidence, ai_prob = predict_with_bombek1(detector, image_path)
            real_prob = 1.0 - ai_prob
            predicted_label = classify_ai_probability(ai_prob)

            writer.writerow(
                {
                    "image_path": str(image_path),
                    "dataset_type": dataset_type,
                    "source_name": source_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "ai_probability": ai_prob,
                    "real_probability": real_prob,
                }
            )

            update_metrics(overall_metrics, true_label, predicted_label, ai_prob)
            update_metrics(source_metrics[(source_name, dataset_type, true_label)], true_label, predicted_label, ai_prob)
            total_rows += 1

            elapsed = time.time() - start_time
            rate = total_rows / elapsed if elapsed > 0 else 0.0
            print(f"Processed {total_rows}/{len(records)} images ({rate:.1f} img/s)", flush=True)

    return total_rows, overall_metrics, source_metrics


def print_summary(summary_rows):
    overall_rows = [row for row in summary_rows if row["source_name"] == "__overall__"]
    if overall_rows:
        row = overall_rows[0]
        print("\nOverall")
        print(f"Total images: {row['total_images']}")
        print(f"Accuracy: {100 * row['accuracy']:.2f}%")
        print(f"Precision (AI): {100 * row['precision_ai']:.2f}%")
        print(f"Recall (AI): {100 * row['recall_ai']:.2f}%")
        print(f"F1 (AI): {100 * row['f1_ai']:.2f}%")
        print(f"False positive rate: {100 * row['false_positive_rate']:.2f}%")
        print(f"False negative rate: {100 * row['false_negative_rate']:.2f}%")
        print(f"Uncertain rate: {100 * row['uncertain_rate']:.2f}%")

    print("\nPer Source")
    for row in sorted((r for r in summary_rows if r["source_name"] != "__overall__"), key=lambda item: (item["dataset_type"], item["source_name"])):
        if row["true_label"] == "ai":
            print(
                f"{row['dataset_type']}/{row['source_name']}: "
                f"n={row['total_images']}, "
                f"detect={100 * row['ai_detect_rate']:.2f}%, "
                f"missed_as_real={100 * row['ai_missed_as_real_rate']:.2f}%, "
                f"uncertain={100 * row['ai_uncertain_rate']:.2f}%, "
                f"avg_ai_prob={100 * row['avg_ai_probability']:.2f}%"
            )
        else:
            print(
                f"{row['dataset_type']}/{row['source_name']}: "
                f"n={row['total_images']}, "
                f"real_accept={100 * row['real_accept_rate']:.2f}%, "
                f"flagged_as_ai={100 * row['real_flagged_as_ai_rate']:.2f}%, "
                f"uncertain={100 * row['real_uncertain_rate']:.2f}%, "
                f"avg_ai_prob={100 * row['avg_ai_probability']:.2f}%"
            )


def main():
    args = parse_args()
    device = resolve_device(args.device)

    real_records = build_records(
        root=args.real_data.resolve(),
        dataset_type="real",
        true_label="real",
        group_mode=args.real_group_mode,
        limit=args.limit_real,
    )
    ai_records = build_records(
        root=args.ai_data.resolve(),
        dataset_type="ai",
        true_label="ai",
        group_mode=args.ai_group_mode,
        limit=args.limit_ai,
    )
    records = real_records + ai_records

    if not records:
        raise FileNotFoundError("No benchmark images were found in the provided roots.")

    effective_num_workers = args.num_workers
    if not can_use_worker_processes(effective_num_workers):
        print(
            "Worker processes are blocked in this environment. Falling back to --num-workers 0.",
            flush=True,
        )
        effective_num_workers = 0

    torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
    print(f"Real images: {len(real_records)}")
    print(f"AI images: {len(ai_records)}")
    print(f"Total images: {len(records)}")
    model = load_backend_model(args.backend, device)

    if args.backend == "bombek1":
        total_rows, overall_metrics, source_metrics = run_inference_bombek1(
            detector=model,
            records=records,
            output_path=args.output.resolve(),
        )
    else:
        print(f"Batch size: {args.batch_size}")
        print(f"Workers: {effective_num_workers}")
        print(f"TTA enabled: {not args.no_tta}")

        loader = build_loader(
            records=records,
            transform=get_eval_transform(),
            batch_size=args.batch_size,
            num_workers=effective_num_workers,
            device=device,
        )

        total_rows, overall_metrics, source_metrics = run_inference(
            model=model,
            loader=loader,
            device=device,
            output_path=args.output.resolve(),
            use_tta=not args.no_tta,
        )

    summary_rows = [
        compute_metric_row(
            source_name="__overall__",
            dataset_type="mixed",
            true_label="mixed",
            metrics=overall_metrics,
        )
    ]
    for (source_name, dataset_type, true_label), metrics in source_metrics.items():
        summary_rows.append(
            compute_metric_row(
                source_name=source_name,
                dataset_type=dataset_type,
                true_label=true_label,
                metrics=metrics,
            )
        )

    write_summary_csv(summary_rows, args.summary_output.resolve())

    print(f"\nSaved predictions to: {args.output.resolve()}")
    print(f"Saved summary to: {args.summary_output.resolve()}")
    print_summary(summary_rows)
    print(f"\nBenchmark complete for {total_rows} images.")


if __name__ == "__main__":
    main()
