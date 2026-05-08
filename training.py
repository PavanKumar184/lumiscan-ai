from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from preprocessing_pipeline import (
    NIH_DISEASE_LABELS,
    DatasetArtifacts,
    build_preprocessing_pipeline,
    create_loss_function,
)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "dr_ultra":
        model = models.drultra_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        "Unsupported model_name. Choose from: 'resnet50', 'densenet121', 'efficientnet_b0'."
    )


def prepare_training_components(
    dataset_root: str,
    model_name: str = "densenet121",
    batch_size: int = 16,
    image_size: int = 224,
) -> Tuple[DatasetArtifacts, nn.Module, nn.Module, torch.optim.Optimizer, torch.device]:
    artifacts = build_preprocessing_pipeline(
        dataset_root=dataset_root,
        batch_size=batch_size,
        image_size=image_size,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        verify_images=False,
        enable_augmentation=True,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name=model_name, num_classes=len(NIH_DISEASE_LABELS)).to(device)
    criterion = create_loss_function(artifacts.bce_pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    return artifacts, model, criterion, optimizer, device


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(dataloader.dataset), 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(dataloader.dataset), 1)


def show_training_setup_summary(
    artifacts: DatasetArtifacts,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, object]:
    summary = {
        "device": str(device),
        "train_samples": len(artifacts.train_df),
        "val_samples": len(artifacts.val_df),
        "test_samples": len(artifacts.test_df),
        "num_classes": len(NIH_DISEASE_LABELS),
        "class_names": NIH_DISEASE_LABELS,
        "bce_pos_weight": artifacts.bce_pos_weight.tolist(),
        "loss_function": str(criterion),
        "optimizer": optimizer.__class__.__name__,
        "model_name": model.__class__.__name__,
    }

    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def main() -> None:
    dataset_root = Path(r"C:\Users\DELL\AppData\Downloads\chest x ray\chest x ray\dataset")
    artifacts, model, criterion, optimizer, device = prepare_training_components(
        dataset_root=str(dataset_root),
        model_name="densenet121",
        batch_size=8,
        image_size=224,
    )

    show_training_setup_summary(
        artifacts=artifacts,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print("\nTraining setup is ready.")
    print("No training loop is started automatically in this file.")
    print("Later, you can call train_one_epoch(...) and validate_one_epoch(...) from your main script.")


if __name__ == "__main__":
    main()
