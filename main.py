import io
import os
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
    resnet50,
)
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# =========================
# CONFIG
# =========================
DATASET_PATH = "genimage"
MODEL_PATH = "best_model.pth"
LAST_CHECKPOINT_PATH = "last_checkpoint.pth"
# Leave these as None to use model-specific presets below.
# You can still override them manually with explicit numeric values.
BATCH_SIZE = None
EPOCHS = None
LR = None
GRAD_ACCUM_STEPS = None
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
NUMBER_OF_WORKERS = 6
PERSISTENT_WORKERS = NUMBER_OF_WORKERS > 0
PREFETCH_FACTOR = 2
IMAGE_SIZE = 224
VAL_RESIZE = 256
MODEL_VARIANT = "dual_stream"
RGB_BACKBONE = "convnext_tiny"
CLASSIFIER_HIDDEN_DIM = 512
FREQUENCY_FEATURE_DIM = 128
MODEL_DROPOUT = 0.30
MAX_GRAD_NORM = 1.0

# Leave one or more generators out to measure unseen-generator generalization.
# Set this to [] only after you have tuned on a hard validation split.
HOLDOUT_GENERATORS = ["sdv"]
# When True, training resumes from LAST_CHECKPOINT_PATH if it exists.
AUTO_RESUME = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =========================
# DATASET + AUGMENTATION
# =========================
class RandomJPEGCompression:
    def __init__(self, quality_range=(35, 95), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        quality = random.randint(*self.quality_range)
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer).convert("RGB")
            return compressed.copy()


class RandomResizeRestore:
    def __init__(self, min_scale=0.45, max_scale=0.9, p=0.35):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        width, height = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        reduced_size = (
            max(32, int(width * scale)),
            max(32, int(height * scale)),
        )
        reduced = image.resize(reduced_size, Image.BILINEAR)
        return reduced.resize((width, height), Image.BILINEAR)


class GenImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        include_generators=None,
        exclude_generators=None,
    ):
        self.samples = []
        self.transform = transform
        self.generators = []
        include_generators = set(include_generators or [])
        exclude_generators = set(exclude_generators or [])

        for generator_name in sorted(os.listdir(root_dir)):
            generator_path = os.path.join(root_dir, generator_name, split)
            if not os.path.isdir(generator_path):
                continue
            if include_generators and generator_name not in include_generators:
                continue
            if generator_name in exclude_generators:
                continue

            self.generators.append(generator_name)

            for class_name in ["ai", "nature"]:
                class_path = os.path.join(generator_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                label = 1 if class_name == "ai" else 0
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    self.samples.append((image_path, label, generator_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, _ = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.85, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)
        ], p=0.5),
        RandomJPEGCompression(),
        RandomResizeRestore(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(VAL_RESIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_available_generators(root_dir):
    return sorted(
        name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )


def build_datasets(root_dir):
    holdout_set = set(HOLDOUT_GENERATORS)
    available_generators = get_available_generators(root_dir)
    missing_generators = sorted(holdout_set - set(available_generators))
    if missing_generators:
        raise ValueError(f"Unknown generators in HOLDOUT_GENERATORS: {missing_generators}")

    if holdout_set and len(holdout_set) == len(available_generators):
        raise ValueError("HOLDOUT_GENERATORS cannot contain every generator.")

    train_dataset = GenImageDataset(
        root_dir=root_dir,
        split="train",
        transform=get_train_transform(),
        exclude_generators=holdout_set,
    )

    val_dataset = GenImageDataset(
        root_dir=root_dir,
        split="val",
        transform=get_eval_transform(),
        include_generators=holdout_set if holdout_set else None,
    )

    return train_dataset, val_dataset


def describe_dataset(name, dataset):
    real_count = sum(1 for _, label, _ in dataset.samples if label == 0)
    ai_count = len(dataset.samples) - real_count
    print(f"{name} samples: {len(dataset)}")
    print(f"{name} generators: {', '.join(dataset.generators)}")
    print(f"{name} class balance -> real: {real_count}, ai: {ai_count}")


def build_loader(dataset, shuffle, batch_size):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUMBER_OF_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    if NUMBER_OF_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(**kwargs)


def get_default_model_config():
    return {
        "variant": MODEL_VARIANT,
        "rgb_backbone": RGB_BACKBONE,
        "hidden_dim": CLASSIFIER_HIDDEN_DIM,
        "frequency_dim": FREQUENCY_FEATURE_DIM,
        "dropout": MODEL_DROPOUT,
        "num_classes": 2,
    }


def get_training_preset(model_config):
    presets = {
        ("single_rgb", "resnet18"): {"batch_size": 160, "epochs": 10, "lr": 3e-4, "grad_accum_steps": 1},
        ("single_rgb", "resnet50"): {"batch_size": 80, "epochs": 14, "lr": 2.25e-4, "grad_accum_steps": 1},
        ("single_rgb", "efficientnet_b0"): {"batch_size": 80, "epochs": 14, "lr": 2.25e-4, "grad_accum_steps": 1},
        ("single_rgb", "convnext_tiny"): {"batch_size": 48, "epochs": 16, "lr": 1.75e-4, "grad_accum_steps": 1},
        ("dual_stream", "resnet18"): {"batch_size": 72, "epochs": 14, "lr": 1.75e-4, "grad_accum_steps": 1},
        ("dual_stream", "resnet50"): {"batch_size": 32, "epochs": 18, "lr": 1.35e-4, "grad_accum_steps": 2},
        ("dual_stream", "efficientnet_b0"): {"batch_size": 40, "epochs": 18, "lr": 1.4e-4, "grad_accum_steps": 2},
        # Tuned for ~8 GB GPUs like the RTX 5060: safe VRAM use with a stronger per-step batch.
        ("dual_stream", "convnext_tiny"): {"batch_size": 48, "epochs": 18, "lr": 1.25e-4, "grad_accum_steps": 1},
    }
    return presets.get(
        (model_config["variant"], model_config["rgb_backbone"]),
        {"batch_size": 48, "epochs": 16, "lr": 1.75e-4, "grad_accum_steps": 1},
    )


def get_training_config(model_config):
    preset = get_training_preset(model_config)
    return {
        "batch_size": preset["batch_size"] if BATCH_SIZE is None else BATCH_SIZE,
        "epochs": preset["epochs"] if EPOCHS is None else EPOCHS,
        "lr": preset["lr"] if LR is None else LR,
        "grad_accum_steps": preset["grad_accum_steps"] if GRAD_ACCUM_STEPS is None else GRAD_ACCUM_STEPS,
    }


def get_legacy_model_config():
    return {
        "variant": "single_rgb",
        "rgb_backbone": "resnet18",
        "hidden_dim": 256,
        "frequency_dim": 0,
        "dropout": 0.20,
        "num_classes": 2,
    }


def get_checkpoint_model_config(checkpoint):
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        return checkpoint["model_config"]
    return get_legacy_model_config()


class ConvNeXtTinyFeatureExtractor(nn.Module):
    def __init__(self, weights):
        super().__init__()
        model = convnext_tiny(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.norm = model.classifier[0]
        self.output_dim = model.classifier[2].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)


def build_rgb_feature_extractor(backbone_name, pretrained):
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        output_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, output_dim

    if backbone_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        output_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, output_dim

    if backbone_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        output_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, output_dim

    if backbone_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = ConvNeXtTinyFeatureExtractor(weights=weights)
        return model, model.output_dim

    raise ValueError(f"Unsupported backbone: {backbone_name}")


class FrequencyArtifactBranch(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        grayscale = x.mean(dim=1, keepdim=True).float()
        spectrum = torch.fft.fft2(grayscale, norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(spectrum))
        magnitude = magnitude / (magnitude.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        features = self.network(magnitude)
        return self.projection(features)


class SingleBackboneClassifier(nn.Module):
    def __init__(self, rgb_encoder, rgb_dim, hidden_dim, dropout, num_classes):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(rgb_dim),
            nn.Dropout(dropout),
            nn.Linear(rgb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.rgb_encoder(x)
        return self.classifier(features)


class DualStreamArtifactDetector(nn.Module):
    def __init__(self, rgb_encoder, rgb_dim, hidden_dim, frequency_dim, dropout, num_classes):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.frequency_encoder = FrequencyArtifactBranch(output_dim=frequency_dim)
        fusion_dim = rgb_dim + frequency_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        rgb_features = self.rgb_encoder(x)
        frequency_features = self.frequency_encoder(x)
        fused_features = torch.cat([rgb_features, frequency_features], dim=1)
        return self.classifier(fused_features)


def build_legacy_resnet18_classifier(pretrained=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_model(pretrained=True, model_config=None):
    config = dict(get_default_model_config() if model_config is None else model_config)
    rgb_encoder, rgb_dim = build_rgb_feature_extractor(
        backbone_name=config["rgb_backbone"],
        pretrained=pretrained,
    )

    if config["variant"] == "single_rgb":
        return SingleBackboneClassifier(
            rgb_encoder=rgb_encoder,
            rgb_dim=rgb_dim,
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            num_classes=config["num_classes"],
        )

    if config["variant"] == "dual_stream":
        return DualStreamArtifactDetector(
            rgb_encoder=rgb_encoder,
            rgb_dim=rgb_dim,
            hidden_dim=config["hidden_dim"],
            frequency_dim=config["frequency_dim"],
            dropout=config["dropout"],
            num_classes=config["num_classes"],
        )

    raise ValueError(f"Unsupported model variant: {config['variant']}")


def build_model_from_checkpoint(checkpoint, pretrained=False):
    if not (isinstance(checkpoint, dict) and "model_config" in checkpoint):
        return build_legacy_resnet18_classifier(pretrained=pretrained)

    model_config = get_checkpoint_model_config(checkpoint)
    return build_model(pretrained=pretrained, model_config=model_config)


def load_trained_model(model_path=MODEL_PATH, device=None):
    target_device = DEVICE if device is None else device
    checkpoint = torch.load(model_path, map_location=target_device)
    model = build_model_from_checkpoint(checkpoint, pretrained=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(target_device)
    model.eval()
    return model, checkpoint


def build_checkpoint(model, epoch, best_acc, optimizer=None, scheduler=None, scaler=None, training_config=None):
    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state": model.state_dict(),
        "model_config": get_default_model_config(),
        "training_config": training_config,
        "holdout_generators": HOLDOUT_GENERATORS,
        "image_size": IMAGE_SIZE,
        "val_resize": VAL_RESIZE,
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()

    return checkpoint


def save_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def load_resume_checkpoint(path, model, optimizer, scheduler, scaler, training_config=None):
    checkpoint = torch.load(path, map_location=DEVICE)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} does not contain resume training state.")

    checkpoint_model_config = get_checkpoint_model_config(checkpoint)
    current_model_config = get_default_model_config()
    if checkpoint_model_config != current_model_config:
        raise ValueError(
            "Checkpoint model configuration does not match the current training model. "
            "Delete or move the old checkpoint before resuming."
        )

    model.load_state_dict(checkpoint["model_state"])

    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    checkpoint_holdout = checkpoint.get("holdout_generators")
    if checkpoint_holdout is not None and checkpoint_holdout != HOLDOUT_GENERATORS:
        print(
            "Warning: checkpoint holdout generators "
            f"{checkpoint_holdout} do not match current setting {HOLDOUT_GENERATORS}."
        )

    checkpoint_training_config = checkpoint.get("training_config")
    if training_config is not None and checkpoint_training_config is not None:
        if checkpoint_training_config != training_config:
            print(
                "Warning: checkpoint training config "
                f"{checkpoint_training_config} does not match current config {training_config}. "
                "Resume will keep optimizer/scheduler state from the checkpoint."
            )

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    return start_epoch, best_acc


# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    epoch_index=None,
    num_epochs=None,
    grad_accum_steps=1,
    max_grad_norm=None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    desc = "Train"
    if epoch_index is not None and num_epochs is not None:
        desc = f"Train {epoch_index+1}/{num_epochs}"

    num_batches = len(loader)
    use_tqdm = tqdm is not None
    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        ascii=True
    ) if use_tqdm else None

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(progress if use_tqdm else loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            raw_loss = criterion(outputs, labels)
            loss = raw_loss / grad_accum_steps

        scaler.scale(loss).backward()

        step_due = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches
        if step_due:
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += raw_loss.item()
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        avg_loss = running_loss / (batch_idx + 1)
        acc = 100 * correct / total if total else 0.0

        if use_tqdm:
            if batch_idx % 10 == 0 or batch_idx + 1 == num_batches:
                progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")
        else:
            print_interval = max(1, num_batches // 20)
            if batch_idx % print_interval == 0 or batch_idx + 1 == num_batches:
                end_char = "\n" if batch_idx + 1 == num_batches else "\r"
                print(
                    f"{desc}: {batch_idx+1}/{num_batches} loss={avg_loss:.4f} acc={acc:.2f}%",
                    end=end_char,
                    flush=True
                )

    if use_tqdm:
        progress.close()

    return running_loss / len(loader), 100 * correct / total


# =========================
# VALIDATION FUNCTION
# =========================
def validate(model, loader, criterion, epoch_index=None, num_epochs=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    desc = "Val"
    if epoch_index is not None and num_epochs is not None:
        desc = f"Val {epoch_index+1}/{num_epochs}"

    num_batches = len(loader)
    use_tqdm = tqdm is not None
    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        ascii=True
    ) if use_tqdm else None

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress if use_tqdm else loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_loss = running_loss / (batch_idx + 1)
            acc = 100 * correct / total if total else 0.0

            if use_tqdm:
                if batch_idx % 10 == 0 or batch_idx + 1 == num_batches:
                    progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")
            else:
                print_interval = max(1, num_batches // 20)
                if batch_idx % print_interval == 0 or batch_idx + 1 == num_batches:
                    end_char = "\n" if batch_idx + 1 == num_batches else "\r"
                    print(
                        f"{desc}: {batch_idx+1}/{num_batches} loss={avg_loss:.4f} acc={acc:.2f}%",
                        end=end_char,
                        flush=True
                    )

    if use_tqdm:
        progress.close()

    return running_loss / len(loader), 100 * correct / total


# =========================
# UPDATED TRAIN ENTRY
# =========================
def run_training():
    model_config = get_default_model_config()
    training_config = get_training_config(model_config)
    print("Number of workers:", NUMBER_OF_WORKERS)
    print("Batch size:", training_config["batch_size"])
    print("Grad accumulation steps:", training_config["grad_accum_steps"])
    print("Effective batch size:", training_config["batch_size"] * training_config["grad_accum_steps"])
    print("Epochs:", training_config["epochs"])
    print("Learning rate:", training_config["lr"])
    print("Max grad norm:", MAX_GRAD_NORM)
    print("Holdout generators:", HOLDOUT_GENERATORS if HOLDOUT_GENERATORS else "None (in-domain validation)")
    print("Model variant:", model_config["variant"])
    print("RGB backbone:", model_config["rgb_backbone"])

    print("\nLoading dataset...")
    train_dataset, val_dataset = build_datasets(DATASET_PATH)
    describe_dataset("Train", train_dataset)
    describe_dataset("Val", val_dataset)

    train_loader = build_loader(train_dataset, shuffle=True, batch_size=training_config["batch_size"])
    val_loader = build_loader(val_dataset, shuffle=False, batch_size=training_config["batch_size"])

    print("\nBuilding model...")
    model = build_model(pretrained=True, model_config=model_config).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=training_config["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    best_acc = 0.0

    if AUTO_RESUME and os.path.exists(LAST_CHECKPOINT_PATH):
        try:
            start_epoch, best_acc = load_resume_checkpoint(
                LAST_CHECKPOINT_PATH,
                model,
                optimizer,
                scheduler,
                scaler,
                training_config=training_config,
            )
            if start_epoch >= training_config["epochs"]:
                print(
                    f"Training already completed through epoch {start_epoch}. "
                    "Delete last_checkpoint.pth or increase EPOCHS to continue."
                )
                return
            print(
                f"Resuming training from epoch {start_epoch + 1}/{training_config['epochs']} "
                f"(best_acc={best_acc:.2f}%)."
            )
        except ValueError as exc:
            print(f"Resume skipped: {exc}")
            print("Starting a fresh run with the current architecture.\n")

    print("\nStarting training...\n")
    for epoch in range(start_epoch, training_config["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            epoch_index=epoch,
            num_epochs=training_config["epochs"],
            grad_accum_steps=training_config["grad_accum_steps"],
            max_grad_norm=MAX_GRAD_NORM,
        )
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            epoch_index=epoch,
            num_epochs=training_config["epochs"],
        )
        scheduler.step()

        print(f"Epoch [{epoch+1}/{training_config['epochs']}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = build_checkpoint(
                model=model,
                epoch=epoch,
                best_acc=best_acc,
                training_config=training_config,
            )
            save_checkpoint(MODEL_PATH, best_checkpoint)
            print("Model saved.\n")

        latest_checkpoint = build_checkpoint(
            model=model,
            epoch=epoch,
            best_acc=best_acc,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            training_config=training_config,
        )
        save_checkpoint(LAST_CHECKPOINT_PATH, latest_checkpoint)

    print("Training complete!")


# =========================
# MAIN
# =========================
def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        x = torch.rand(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        print("Test Tensor Device:", y.device)

    run_training()
    return

    print("Number of workers:", NUMBER_OF_WORKERS)
    print("Batch size:", BATCH_SIZE)
    print("Holdout generators:", HOLDOUT_GENERATORS if HOLDOUT_GENERATORS else "None (in-domain validation)")

    print("\nLoading dataset...")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = GenImageDataset(DATASET_PATH, "train", train_transform)
    val_dataset   = GenImageDataset(DATASET_PATH, "val", val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ⚠️ IMPORTANT: num_workers = 0 first (stable)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )

    print("\nBuilding model...")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            epoch_index=epoch, num_epochs=EPOCHS
        )
        val_acc = validate(model, val_loader, epoch_index=epoch, num_epochs=EPOCHS)

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model saved!\n")

    print("Training Complete!")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
