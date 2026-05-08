from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


NIH_DISEASE_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


@dataclass
class DatasetArtifacts:
    metadata: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    class_counts: pd.Series
    class_frequencies: pd.Series
    inverse_frequency_weights: torch.Tensor
    bce_pos_weight: torch.Tensor


EXPORT_METADATA_COLUMNS: List[str] = [
    "Image Index",
    "Finding Labels",
    "image_path",
    "label_signature",
    "num_labels",
    "has_finding",
    *NIH_DISEASE_LABELS,
]


class NIHChestXrayDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_columns: Sequence[str],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.label_columns = list(label_columns)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = Path(row["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        return image, labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_csv_path(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "Data_Entry_2017.csv" / "Data_Entry_2017.csv",
        dataset_root / "Data_Entry_2017.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Data_Entry_2017.csv in the provided dataset root."
    )


def build_image_index(dataset_root: Path) -> Dict[str, str]:
    image_index: Dict[str, str] = {}
    for image_dir in sorted(dataset_root.glob("images_*")):
        nested_dir = image_dir / "images"
        if not nested_dir.exists():
            continue
        for image_path in nested_dir.glob("*.png"):
            image_index[image_path.name] = str(image_path.resolve())
    return image_index


def normalize_labels(label_string: str) -> List[str]:
    if pd.isna(label_string):
        return []

    labels = [label.strip() for label in str(label_string).split("|") if label.strip()]
    labels = [label for label in labels if label != "No Finding"]

    unknown_labels = sorted(set(labels) - set(NIH_DISEASE_LABELS))
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")
    return labels


def is_valid_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def encode_multi_hot(labels: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    encoded = np.zeros(len(class_names), dtype=np.float32)
    label_to_index = {label: index for index, label in enumerate(class_names)}
    for label in labels:
        encoded[label_to_index[label]] = 1.0
    return encoded


def load_and_clean_metadata(
    dataset_root: Path,
    class_names: Sequence[str] = NIH_DISEASE_LABELS,
    verify_images: bool = False,
) -> pd.DataFrame:
    csv_path = resolve_csv_path(dataset_root)
    metadata = pd.read_csv(csv_path)

    required_columns = {"Image Index", "Finding Labels"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    image_index = build_image_index(dataset_root)
    metadata["image_path"] = metadata["Image Index"].map(image_index)
    metadata = metadata.dropna(subset=["image_path"]).copy()

    metadata["label_list"] = metadata["Finding Labels"].apply(normalize_labels)
    metadata["num_labels"] = metadata["label_list"].apply(len)
    metadata["has_finding"] = (metadata["num_labels"] > 0).astype(np.int64)

    encoded_labels = np.vstack(
        metadata["label_list"].apply(lambda labels: encode_multi_hot(labels, class_names))
    )
    encoded_df = pd.DataFrame(encoded_labels, columns=class_names, index=metadata.index)
    metadata = pd.concat([metadata, encoded_df], axis=1)

    if verify_images:
        metadata["is_valid_image"] = metadata["image_path"].apply(is_valid_image)
        metadata = metadata[metadata["is_valid_image"]].copy()
        metadata = metadata.drop(columns=["is_valid_image"])

    metadata["label_signature"] = metadata["label_list"].apply(
        lambda labels: "|".join(sorted(labels)) if labels else "No Finding"
    )
    metadata = metadata.reset_index(drop=True)
    return metadata


def compute_class_weights(
    dataframe: pd.DataFrame,
    class_names: Sequence[str] = NIH_DISEASE_LABELS,
) -> Tuple[pd.Series, pd.Series, torch.Tensor, torch.Tensor]:
    label_matrix = dataframe[list(class_names)].to_numpy(dtype=np.float32)
    num_samples = float(len(dataframe))
    positive_counts = pd.Series(label_matrix.sum(axis=0), index=class_names, dtype=np.float32)
    class_frequencies = positive_counts / max(num_samples, 1.0)

    inverse_frequency = 1.0 / np.clip(class_frequencies.to_numpy(dtype=np.float32), 1e-8, None)
    inverse_frequency = inverse_frequency / inverse_frequency.mean()

    negatives = num_samples - positive_counts.to_numpy(dtype=np.float32)
    positives = np.clip(positive_counts.to_numpy(dtype=np.float32), 1.0, None)
    bce_pos_weight = negatives / positives

    return (
        positive_counts,
        class_frequencies,
        torch.tensor(inverse_frequency, dtype=torch.float32),
        torch.tensor(bce_pos_weight, dtype=torch.float32),
    )


def _split_sizes(num_samples: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train_size = int(round(num_samples * train_ratio))
    val_size = int(round(num_samples * val_ratio))
    test_size = num_samples - train_size - val_size

    if test_size < 0:
        raise ValueError("Split ratios are invalid. They must sum to <= 1.0.")
    return train_size, val_size, test_size


def multilabel_stratified_split(
    dataframe: pd.DataFrame,
    class_names: Sequence[str] = NIH_DISEASE_LABELS,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    rng = np.random.default_rng(seed)
    label_matrix = dataframe[list(class_names)].to_numpy(dtype=np.int64)
    num_samples = len(dataframe)
    train_size, val_size, test_size = _split_sizes(num_samples, train_ratio, val_ratio)

    split_names = ["train", "val", "test"]
    target_sizes = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    current_sizes = {name: 0 for name in split_names}
    target_label_counts = {
        "train": label_matrix.sum(axis=0) * train_ratio,
        "val": label_matrix.sum(axis=0) * val_ratio,
        "test": label_matrix.sum(axis=0) * (1.0 - train_ratio - val_ratio),
    }
    current_label_counts = {
        name: np.zeros(len(class_names), dtype=np.float64) for name in split_names
    }
    assigned_indices = {name: [] for name in split_names}

    positive_cardinality = label_matrix.sum(axis=1)
    noise = rng.random(num_samples)
    order = np.lexsort((noise, -positive_cardinality))

    for sample_index in order:
        sample_labels = label_matrix[sample_index]
        positive_positions = np.where(sample_labels == 1)[0]

        available_splits = [
            name for name in split_names if current_sizes[name] < target_sizes[name]
        ]
        if not available_splits:
            break

        best_split = None
        best_score = None

        for split_name in available_splits:
            size_deficit = target_sizes[split_name] - current_sizes[split_name]
            label_deficit = target_label_counts[split_name] - current_label_counts[split_name]
            label_deficit = np.clip(label_deficit, a_min=0.0, a_max=None)

            if len(positive_positions) > 0:
                score = float(label_deficit[positive_positions].sum())
            else:
                score = 0.0

            score += float(size_deficit) * 0.05

            if best_score is None or score > best_score:
                best_score = score
                best_split = split_name

        assert best_split is not None
        assigned_indices[best_split].append(sample_index)
        current_sizes[best_split] += 1
        current_label_counts[best_split] += sample_labels

    train_df = dataframe.iloc[assigned_indices["train"]].sample(frac=1.0, random_state=seed)
    val_df = dataframe.iloc[assigned_indices["val"]].sample(frac=1.0, random_state=seed)
    test_df = dataframe.iloc[assigned_indices["test"]].sample(frac=1.0, random_state=seed)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_train_transforms(image_size: int = 224, enable_augmentation: bool = True) -> transforms.Compose:
    transform_steps: List[transforms.Compose] = [transforms.Resize((image_size, image_size))]
    if enable_augmentation:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(transform_steps)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str] = NIH_DISEASE_LABELS,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    enable_augmentation: bool = True,
) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)

    train_dataset = NIHChestXrayDataset(
        dataframe=train_df,
        label_columns=class_names,
        transform=get_train_transforms(image_size=image_size, enable_augmentation=enable_augmentation),
    )
    val_dataset = NIHChestXrayDataset(
        dataframe=val_df,
        label_columns=class_names,
        transform=get_eval_transforms(image_size=image_size),
    )
    test_dataset = NIHChestXrayDataset(
        dataframe=test_df,
        label_columns=class_names,
        transform=get_eval_transforms(image_size=image_size),
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def print_dataset_statistics(
    dataframe: pd.DataFrame,
    class_counts: pd.Series,
    class_frequencies: pd.Series,
    inverse_frequency_weights: torch.Tensor,
    bce_pos_weight: torch.Tensor,
) -> None:
    print("\n========== DATASET STATISTICS ==========")
    print(f"Total cleaned samples: {len(dataframe)}")
    print(f"No Finding samples: {(dataframe['num_labels'] == 0).sum()}")
    print(f"Samples with one or more diseases: {(dataframe['num_labels'] > 0).sum()}")
    print(f"Average labels per image: {dataframe['num_labels'].mean():.3f}")

    stats_df = pd.DataFrame(
        {
            "positive_count": class_counts.astype(int),
            "frequency": class_frequencies.round(6),
            "inverse_frequency_weight": inverse_frequency_weights.numpy().round(6),
            "bce_pos_weight": bce_pos_weight.numpy().round(6),
        }
    )
    print("\nPer-class distribution on the training split and the resulting weights:")
    print(stats_df.to_string())
    print("========================================\n")


def save_split_metadata_csvs(
    output_dir: str | Path,
    metadata: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    export_columns = [
        column for column in EXPORT_METADATA_COLUMNS if column in metadata.columns
    ]

    files = {
        "cleaned_metadata": output_path / "cleaned_metadata.csv",
        "train_metadata": output_path / "train_metadata.csv",
        "val_metadata": output_path / "val_metadata.csv",
        "test_metadata": output_path / "test_metadata.csv",
    }

    metadata.loc[:, export_columns].to_csv(files["cleaned_metadata"], index=False)
    train_df.loc[:, export_columns].to_csv(files["train_metadata"], index=False)
    val_df.loc[:, export_columns].to_csv(files["val_metadata"], index=False)
    test_df.loc[:, export_columns].to_csv(files["test_metadata"], index=False)
    return files


def save_class_weights_csv(
    output_dir: str | Path,
    class_counts: pd.Series,
    class_frequencies: pd.Series,
    inverse_frequency_weights: torch.Tensor,
    bce_pos_weight: torch.Tensor,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_df = pd.DataFrame(
        {
            "class_name": class_counts.index.tolist(),
            "positive_count": class_counts.astype(int).tolist(),
            "frequency": class_frequencies.astype(float).tolist(),
            "inverse_frequency_weight": inverse_frequency_weights.cpu().numpy().tolist(),
            "bce_pos_weight": bce_pos_weight.cpu().numpy().tolist(),
        }
    )

    weights_path = output_path / "class_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    return weights_path


def build_preprocessing_pipeline(
    dataset_root: str,
    batch_size: int = 32,
    image_size: int = 224,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    verify_images: bool = False,
    enable_augmentation: bool = True,
    num_workers: Optional[int] = None,
) -> DatasetArtifacts:
    set_seed(seed)
    dataset_root_path = Path(dataset_root)

    metadata = load_and_clean_metadata(
        dataset_root=dataset_root_path,
        class_names=NIH_DISEASE_LABELS,
        verify_images=verify_images,
    )

    train_df, val_df, test_df = multilabel_stratified_split(
        dataframe=metadata,
        class_names=NIH_DISEASE_LABELS,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    class_counts, class_frequencies, inverse_frequency_weights, bce_pos_weight = compute_class_weights(
        dataframe=train_df,
        class_names=NIH_DISEASE_LABELS,
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_names=NIH_DISEASE_LABELS,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        enable_augmentation=enable_augmentation,
    )

    return DatasetArtifacts(
        metadata=metadata,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_counts=class_counts,
        class_frequencies=class_frequencies,
        inverse_frequency_weights=inverse_frequency_weights,
        bce_pos_weight=bce_pos_weight,
    )


def create_loss_function(bce_pos_weight: torch.Tensor) -> torch.nn.Module:
    return torch.nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 preprocessing pipeline.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing Data_Entry_2017.csv and image folders.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open each image once during metadata cleaning to remove corrupt files.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable training-time augmentation.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Optional directory where cleaned and split metadata CSVs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_preprocessing_pipeline(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        verify_images=args.verify_images,
        enable_augmentation=not args.disable_augmentation,
        num_workers=args.num_workers,
    )

    print_dataset_statistics(
        dataframe=artifacts.metadata,
        class_counts=artifacts.class_counts,
        class_frequencies=artifacts.class_frequencies,
        inverse_frequency_weights=artifacts.inverse_frequency_weights,
        bce_pos_weight=artifacts.bce_pos_weight,
    )

    print(f"Train samples: {len(artifacts.train_df)}")
    print(f"Validation samples: {len(artifacts.val_df)}")
    print(f"Test samples: {len(artifacts.test_df)}")

    images, labels = next(iter(artifacts.train_loader))
    print(f"One train batch image tensor shape: {tuple(images.shape)}")
    print(f"One train batch label tensor shape: {tuple(labels.shape)}")

    criterion = create_loss_function(artifacts.bce_pos_weight)
    print(f"Loss function ready: {criterion}")

    if args.export_dir:
        exported_files = save_split_metadata_csvs(
            output_dir=args.export_dir,
            metadata=artifacts.metadata,
            train_df=artifacts.train_df,
            val_df=artifacts.val_df,
            test_df=artifacts.test_df,
        )
        print("\nExported metadata files:")
        for name, path in exported_files.items():
            print(f"{name}: {path}")

        weights_path = save_class_weights_csv(
            output_dir=args.export_dir,
            class_counts=artifacts.class_counts,
            class_frequencies=artifacts.class_frequencies,
            inverse_frequency_weights=artifacts.inverse_frequency_weights,
            bce_pos_weight=artifacts.bce_pos_weight,
        )
        print(f"class_weights: {weights_path}")


if __name__ == "__main__":
    main()
