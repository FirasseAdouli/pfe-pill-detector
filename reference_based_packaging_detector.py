import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

LABEL_MAP = {
    "conform": 0,
    "non_conform": 1,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def normalize_label(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def get_device(force_cuda: bool = False):
    if force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was forced, but no CUDA-capable PyTorch GPU is available.\n"
                "Check your NVIDIA driver and install a CUDA-enabled PyTorch build."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv_folder_pattern(csv_path: Path) -> Tuple[str, str]:
    """
    Example:
      03-03.csv -> ("03", "03")
      12-07.csv -> ("12", "07")
    """
    stem = csv_path.stem
    parts = stem.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"CSV filename '{csv_path.name}' does not match expected pattern 'XX-YY.csv'"
        )
    return parts[0], parts[1]


def find_image_file(folder: Path, filename: str) -> Optional[Path]:
    """
    Tries exact filename first.
    If not found, tries matching same stem with known image extensions.
    """
    candidate = folder / filename
    if candidate.exists():
        return candidate

    stem = Path(filename).stem
    for ext in IMAGE_EXTS:
        alt = folder / f"{stem}{ext}"
        if alt.exists():
            return alt

    return None


def load_dataset_from_csv_folder(dataset_dir: str, project_root: str) -> pd.DataFrame:
    """
    Reads every CSV under dataset_dir.
    For dataset/03-03.csv, images are expected under project_root/03/03/
    """
    dataset_dir = Path(dataset_dir)
    project_root = Path(project_root)

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    all_rows = []
    missing_images = []

    for csv_file in csv_files:
        root_name, child_name = parse_csv_folder_pattern(csv_file)
        image_folder = project_root / root_name / child_name

        if not image_folder.exists():
            raise FileNotFoundError(
                f"Expected image folder '{image_folder}' for CSV '{csv_file.name}', but it does not exist."
            )

        df = pd.read_csv(csv_file)

        required_cols = {"filename", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV '{csv_file.name}' must contain columns {required_cols}, missing: {missing_cols}"
            )

        df = df.copy()
        df["filename"] = df["filename"].astype(str).str.strip()
        df["label"] = df["label"].apply(normalize_label)

        unknown = sorted(set(df["label"]) - set(LABEL_MAP.keys()))
        if unknown:
            raise ValueError(
                f"CSV '{csv_file.name}' has unknown labels {unknown}. Allowed: {list(LABEL_MAP.keys())}"
            )

        for _, row in df.iterrows():
            filename = row["filename"]
            label = row["label"]
            image_path = find_image_file(image_folder, filename)

            if image_path is None:
                missing_images.append(f"{csv_file.name} -> {image_folder} -> {filename}")
                continue

            all_rows.append({
                "csv_file": csv_file.name,
                "filename": filename,
                "label": label,
                "label_id": LABEL_MAP[label],
                "image_path": str(image_path),
            })

    if missing_images:
        preview = missing_images[:15]
        raise FileNotFoundError(
            f"{len(missing_images)} image(s) referenced in CSV files were not found.\n"
            f"Examples:\n" + "\n".join(preview)
        )

    if not all_rows:
        raise RuntimeError("No valid image rows were loaded from the dataset.")

    return pd.DataFrame(all_rows)


class TrainingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label_id"])

        if self.transform:
            image = self.transform(image)

        return image, label


class PredictFolderDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(image_path)


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc, all_true, all_pred


def train_model(
    dataset_dir: str,
    project_root: str,
    output_model: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_size: float = 0.2,
    force_cuda: bool = False,
):
    df = load_dataset_from_csv_folder(dataset_dir, project_root)

    if len(df) < 2:
        raise ValueError("Need at least 2 samples to train.")

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=42,
        stratify=df["label_id"],
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = TrainingDataset(train_df, transform=train_transform)
    val_dataset = TrainingDataset(val_df, transform=val_transform)

    use_cuda = get_device(force_cuda=force_cuda).type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    device = get_device(force_cuda=force_cuda)
    model = build_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    best_y_true = None
    best_y_pred = None

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_y_true = y_true
            best_y_pred = y_pred
            best_state = {
                "model_state_dict": model.state_dict(),
                "label_map": LABEL_MAP,
                "arch": "resnet18",
                "img_size": 224,
            }

    if best_state is None:
        raise RuntimeError("Training failed: no model state captured.")

    torch.save(best_state, output_model)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model to: {output_model}")

    if best_y_true is not None and best_y_pred is not None:
        print("\nClassification report:")
        print(classification_report(best_y_true, best_y_pred, target_names=["conform", "non_conform"]))
        print("Confusion matrix:")
        print(confusion_matrix(best_y_true, best_y_pred))


def load_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_infer_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict_one(model_path: str, image_path: str, force_cuda: bool = False):
    device = get_device(force_cuda=force_cuda)
    model = load_model(model_path, device)
    transform = get_infer_transform()

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    label = IDX_TO_LABEL[pred_idx]
    confidence = float(probs[pred_idx].item())

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


def batch_predict(model_path: str, folder: str, output_csv: Optional[str] = None, force_cuda: bool = False):
    device = get_device(force_cuda=force_cuda)
    model = load_model(model_path, device)
    transform = get_infer_transform()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    folder = Path(folder)
    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in folder: {folder}")

    use_cuda = device.type == "cuda"

    dataset = PredictFolderDataset(image_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    rows = []

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(paths)):
                pred_idx = int(preds[i].item())
                label = IDX_TO_LABEL[pred_idx]
                confidence = float(probs[i, pred_idx].item())

                rows.append({
                    "image_path": paths[i],
                    "prediction": label,
                    "confidence": confidence,
                })

                print(f"{Path(paths[i]).name}: {label} | {confidence:.2%}")

    if output_csv:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Wrote predictions to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Train conform/non_conform classifier from dataset CSV naming convention"
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train using all CSV files in dataset/")
    p_train.add_argument("--dataset-dir", default="dataset", help="Folder containing CSV files like 03-03.csv")
    p_train.add_argument("--project-root", default=".", help="Root folder containing image directories like 03/03/")
    p_train.add_argument("--output-model", default="conform_classifier.pth")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--val-size", type=float, default=0.2)
    p_train.add_argument("--force-cuda", action="store_true", help="Require CUDA GPU")

    p_predict = sub.add_parser("predict", help="Predict one image")
    p_predict.add_argument("--model", required=True)
    p_predict.add_argument("--image", required=True)
    p_predict.add_argument("--force-cuda", action="store_true", help="Require CUDA GPU")

    p_batch = sub.add_parser("batch", help="Predict all images in a folder")
    p_batch.add_argument("--model", required=True)
    p_batch.add_argument("--folder", required=True)
    p_batch.add_argument("--output-csv")
    p_batch.add_argument("--force-cuda", action="store_true", help="Require CUDA GPU")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            dataset_dir=args.dataset_dir,
            project_root=args.project_root,
            output_model=args.output_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_size=args.val_size,
            force_cuda=args.force_cuda,
        )
    elif args.mode == "predict":
        predict_one(args.model, args.image, force_cuda=args.force_cuda)
    elif args.mode == "batch":
        batch_predict(args.model, args.folder, args.output_csv, force_cuda=args.force_cuda)


if __name__ == "__main__":
    main()