"""
Face Recognition for Autonomous Drone Tracking
================================================
Vision Transformer (ViT) based face recognition system using ArcFace loss.
Designed as the perception module of an intelligent "Follow Me" drone system.

Architecture:
    - Backbone  : ViT-Base/16 (pretrained on ImageNet via timm)
    - Head      : Linear projection to 512-d L2-normalized embedding space
    - Loss      : ArcFace (Additive Angular Margin Loss)
    - Inference : Cosine similarity against a pre-built identity gallery

Usage:
    1. Set paths in CONFIG (data_dir, gallery_dir, output paths)
    2. Run: python train.py
    3. Trained model saved as face_vit_model_best.pth

Author : <Your Name>
Date   : 2025
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import math
import os
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: Dict = {
    # ── Paths (edit these before running) ─────────────────────────────────────
    "data_dir":                    "/kaggle/input/recognition/DATASET IMAGES",
    "gallery_dir":                 "/kaggle/input/recognition/DATASET IMAGES",
    "model_save_path":             "/kaggle/working/face_vit_model_best.pth",
    "criterion_save_path":         "/kaggle/working/face_vit_model_best_criterion.pth",
    "history_plot_save_path":      "/kaggle/working/training_history.png",
    "eval_curves_plot_save_path":  "/kaggle/working/evaluation_curves.png",
    "confusion_matrix_plot_save_path": "/kaggle/working/confusion_matrix.png",
    "haar_cascade_path":           "/kaggle/working/haarcascade_frontalface_default.xml",

    # ── Model ──────────────────────────────────────────────────────────────────
    "img_size":        224,
    "patch_size":      16,
    "embedding_dim":   512,
    "vit_model_name":  "vit_base_patch16_224",
    "pretrained":      True,

    # ── Training ───────────────────────────────────────────────────────────────
    "learning_rate":      1e-4,
    "weight_decay":       0.01,
    "batch_size":         32,
    "num_epochs":         25,
    "validation_split":   0.2,
    "num_workers":        2,

    # ── ArcFace Loss ───────────────────────────────────────────────────────────
    "arcface_s":  30.0,   # Scale factor
    "arcface_m":  0.50,   # Angular margin (radians)

    # ── Inference ─────────────────────────────────────────────────────────────
    "recognition_threshold": 0.60,

    # ── Auto-populated at runtime ─────────────────────────────────────────────
    "num_classes":   None,
    "class_names":   None,
    "class_to_idx":  None,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs("/kaggle/working/", exist_ok=True)
print(f"Device       : {CONFIG['device']}")
print(f"Dataset path : {CONFIG['data_dir']}")
print(f"Model output : {CONFIG['model_save_path']}")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def download_haar_cascade(save_path: str) -> bool:
    """
    Download the OpenCV frontal-face Haar Cascade XML if not already present.

    Args:
        save_path: Destination file path for the XML file.

    Returns:
        True if the file is available (downloaded or pre-existing), else False.
    """
    if os.path.exists(save_path):
        print("Haar Cascade already exists — skipping download.")
        return True

    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master"
        "/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    print(f"Downloading Haar Cascade -> {save_path}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as exc:
        print(f"Download failed: {exc}")
        return False


def get_inference_transform(img_size: int) -> transforms.Compose:
    """
    Return the standard ImageNet normalisation transform for inference.
    No data augmentation is applied.

    Args:
        img_size: Target square image size (pixels).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(
    config: Dict,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Build training and validation DataLoaders from an ImageFolder dataset.

    Expected dataset layout::

        data_dir/
            person_A/  img1.jpg  img2.jpg  ...
            person_B/  img1.jpg  ...

    Augmentation (random flip, rotation, colour jitter) is applied to the
    training split only. Validation uses the clean inference transform.

    Args:
        config: Global CONFIG dict. Updates num_classes, class_names,
                and class_to_idx in place.

    Returns:
        Tuple of (train_loader, val_loader). Either may be None on failure.
    """
    img_size         = config["img_size"]
    batch_size       = config["batch_size"]
    validation_split = config["validation_split"]
    num_workers      = config["num_workers"]
    data_dir         = config["data_dir"]

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = get_inference_transform(img_size)

    # Load dataset and populate config metadata
    try:
        probe_ds = ImageFolder(data_dir)
        if not probe_ds.classes:
            print(f"No class sub-directories found in {data_dir!r}.")
            return None, None

        config["num_classes"]  = len(probe_ds.classes)
        config["class_names"]  = probe_ds.classes
        config["class_to_idx"] = probe_ds.class_to_idx
        print(f"Dataset: {len(probe_ds)} images | {config['num_classes']} classes")
        print(f"Classes: {probe_ds.classes}")

        train_ds = ImageFolder(data_dir, transform=train_transform)
        val_ds   = ImageFolder(data_dir, transform=val_transform)

    except FileNotFoundError:
        print(f"Dataset directory not found: {data_dir!r}")
        return None, None

    # Random index split
    total   = len(train_ds)
    indices = list(range(total))
    np.random.shuffle(indices)
    n_val        = int(np.floor(validation_split * total))
    train_idx    = indices[n_val:]
    val_idx      = indices[:n_val]

    print(f"Train samples : {len(train_idx)}")
    print(f"Val   samples : {len(val_idx)}")

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = (
        DataLoader(
            Subset(val_ds, val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        if n_val > 0 else None
    )

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FaceViT(nn.Module):
    """
    Vision Transformer backbone with a projection head for face embedding.

    The backbone (ViT-Base/16) is loaded from timm with its classification
    head removed (num_classes=0). A linear layer projects the [CLS] token
    output to a 512-d embedding space, which is L2-normalised — required
    for both ArcFace training and cosine-similarity inference.

    Args:
        model_name:    timm model identifier (e.g. "vit_base_patch16_224").
        pretrained:    Whether to load ImageNet weights for the backbone.
        embedding_dim: Output embedding dimensionality.
        img_size:      Input image resolution (used for feature-dim inference).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        embedding_dim: int,
        img_size: int,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Infer backbone output dimension
        try:
            feat_dim = self.backbone.num_features
        except AttributeError:
            with torch.no_grad():
                feat_dim = self.backbone(torch.randn(1, 3, img_size, img_size)).shape[-1]

        self.embedding_head = nn.Linear(feat_dim, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_head.weight)
        nn.init.constant_(self.embedding_head.bias, 0)

        print(f"FaceViT | backbone={model_name}  pretrained={pretrained}")
        print(f"         feat_dim={feat_dim} -> embedding_dim={embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone -> linear projection -> L2 normalisation.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            L2-normalised face embeddings of shape (B, embedding_dim).
        """
        features   = self.backbone(x)
        embeddings = self.embedding_head(features)
        return F.normalize(embeddings, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for face recognition.

    Introduces an angular margin m between query embeddings and target class
    centres, maximising inter-class separability on the unit hypersphere.

    Reference:
        Deng et al. (2019) — "ArcFace: Additive Angular Margin Loss for
        Deep Face Recognition." CVPR 2019.

    Args:
        in_features:  Input embedding dimension (e.g. 512).
        out_features: Number of identity classes.
        s:            Feature scale factor (default 30.0).
        m:            Angular margin in radians (default 0.50).
        easy_margin:  Use simplified margin boundary (default False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.s           = s
        self.m           = m
        self.easy_margin = easy_margin

        # Learnable class-centre weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-computed margin constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)      # Stability threshold
        self.mm    = math.sin(math.pi - m) * m  # Penalty fallback

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: L2-normalised embeddings (B, in_features).
            labels:     Ground-truth class indices (B,).

        Returns:
            Scalar cross-entropy loss with angular margin applied.
        """
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-6))
        phi    = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, labels)

    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Return scaled cosine-similarity logits (no margin) for evaluation.

        Args:
            embeddings: L2-normalised embeddings (B, in_features).

        Returns:
            Logit tensor (B, num_classes).
        """
        with torch.no_grad():
            cosine = F.linear(embeddings, F.normalize(self.weight))
        return cosine * self.s


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    criterion: ArcFaceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict,
) -> Dict:
    """
    Train FaceViT, tracking the best checkpoint by validation accuracy.

    Saves two files whenever a new best is reached:
        config['model_save_path']    — model state dict
        config['criterion_save_path'] — ArcFace weight state dict

    Args:
        model:        FaceViT model on target device.
        criterion:    ArcFaceLoss on target device.
        optimizer:    AdamW covering both model and criterion parameters.
        scheduler:    CosineAnnealingLR scheduler.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split (may be None).
        config:       Global CONFIG dict.

    Returns:
        History dict with keys train_loss, val_loss, val_acc, val_auc.
    """
    device     = config["device"]
    num_epochs = config["num_epochs"]

    best_val_acc = 0.0
    best_epoch   = -1
    history      = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    print(f"\n{'='*60}")
    print(f"  Training | epochs={num_epochs}  lr={config['learning_rate']}  "
          f"batch={config['batch_size']}  device={device}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()

        # Training step
        model.train()
        criterion.train()
        running_loss = n_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_train      += images.size(0)

        epoch_train_loss = running_loss / n_train if n_train else 0.0
        history["train_loss"].append(epoch_train_loss)

        # Validation step
        epoch_val_loss = epoch_val_acc = epoch_val_auc = 0.0
        if val_loader:
            val_metrics    = evaluate_model(model, criterion, val_loader, device,
                                            config["num_classes"], is_eval=True)
            epoch_val_loss = val_metrics.get("loss", 0.0)
            epoch_val_acc  = val_metrics.get("accuracy", 0.0)
            epoch_val_auc  = val_metrics.get("roc_auc_micro", 0.0)

        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_auc"].append(epoch_val_auc)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}] ({time.time()-t0:.1f}s) | "
            f"lr={current_lr:.2e} | "
            f"train_loss={epoch_train_loss:.4f} | "
            f"val_loss={epoch_val_loss:.4f} | "
            f"val_acc={epoch_val_acc:.4f} | "
            f"val_auc={epoch_val_auc:.4f}"
        )

        # Save checkpoint if validation accuracy improved
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch   = epoch + 1
            torch.save(model.state_dict(),     config["model_save_path"])
            torch.save(criterion.state_dict(), config["criterion_save_path"])
            print(f"  -> New best val_acc={best_val_acc:.4f} — checkpoint saved.")

    print(f"\nTraining complete.  Best val_acc={best_val_acc:.4f} at epoch {best_epoch}.")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: nn.Module,
    criterion: ArcFaceLoss,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    is_eval: bool = False,
) -> Dict:
    """
    Compute comprehensive evaluation metrics on a given dataloader.

    Metrics returned:
        loss, accuracy, overall_fpr, precision_weighted, recall_weighted,
        f1_weighted, roc_auc_micro, avg_precision_micro, mAP_macro,
        fpr_micro, tpr_micro, precision_micro, recall_micro, confusion_matrix.

    Args:
        model:       FaceViT in eval mode.
        criterion:   ArcFaceLoss with loaded weights.
        dataloader:  DataLoader to evaluate on.
        device:      Torch device string.
        num_classes: Total number of identity classes.
        is_eval:     If False, prints a detailed metrics summary to stdout.

    Returns:
        Dict of computed metrics.
    """
    model.eval()
    criterion.eval()

    all_labels, all_preds, all_scores = [], [], []
    running_loss = total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            logits     = criterion.get_logits(embeddings)
            running_loss += F.cross_entropy(logits, labels).item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_scores.append(logits.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            total += labels.size(0)

    if total == 0:
        return {}

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_scores = np.concatenate(all_scores, axis=0)
    labels_bin = label_binarize(all_labels, classes=range(num_classes))

    metrics: Dict = {"loss": running_loss / total}
    metrics["accuracy"] = float(np.mean(all_preds == all_labels))

    # Per-class and aggregate classification metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    metrics.update(precision_weighted=prec, recall_weighted=rec, f1_weighted=f1)

    # ROC / PR curves (micro-average across all classes)
    fpr_m, tpr_m, _ = roc_curve(labels_bin.ravel(), all_scores.ravel())
    metrics["roc_auc_micro"] = auc(fpr_m, tpr_m)
    metrics["fpr_micro"]     = fpr_m
    metrics["tpr_micro"]     = tpr_m

    prec_m, rec_m, _ = precision_recall_curve(labels_bin.ravel(), all_scores.ravel())
    metrics["avg_precision_micro"] = average_precision_score(labels_bin, all_scores, average="micro")
    metrics["precision_micro"]     = prec_m
    metrics["recall_micro"]        = rec_m

    try:
        metrics["mAP_macro"] = average_precision_score(labels_bin, all_scores, average="macro")
    except ValueError:
        metrics["mAP_macro"] = 0.0

    # Confusion matrix and overall FPR derived from it
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    metrics["confusion_matrix"] = cm

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    total_FP, total_TN   = FP.sum(), TN.sum()
    metrics["overall_fpr"] = total_FP / (total_FP + total_TN) if (total_FP + total_TN) else 0.0

    if not is_eval:
        print("\n── Evaluation Results ───────────────────────────────────────────")
        for key, fmt in [
            ("loss",                ".4f"),
            ("accuracy",            ".4f"),
            ("overall_fpr",         ".4f"),
            ("precision_weighted",  ".4f"),
            ("recall_weighted",     ".4f"),
            ("f1_weighted",         ".4f"),
            ("roc_auc_micro",       ".4f"),
            ("avg_precision_micro", ".4f"),
            ("mAP_macro",           ".4f"),
        ]:
            print(f"  {key:<26} {metrics[key]:{fmt}}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history: Dict, save_path: str) -> None:
    """
    Plot and save training / validation loss, accuracy, and AUC curves.

    Args:
        history:   Dict produced by train_model.
        save_path: Output PNG file path.
    """
    if not history or not history.get("train_loss"):
        print("Nothing to plot — history is empty.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 12))

    axes[0].plot(epochs, history["train_loss"], "bo-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "ro-", label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["val_acc"], "go-", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, history["val_auc"], "mo-", label="Val Micro-AUC")
    axes[2].set_title("Validation ROC AUC (Micro)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC")
    axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Training history saved -> {save_path}")


def plot_evaluation_curves(metrics: Dict, config: Dict) -> None:
    """
    Plot and save ROC curve, Precision-Recall curve, and confusion matrix.

    Args:
        metrics: Dict produced by evaluate_model.
        config:  Global CONFIG (provides save paths and class names).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr = metrics.get("fpr_micro", np.array([])), metrics.get("tpr_micro", np.array([]))
    ax        = axes[0]
    if fpr.size:
        ax.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"Micro-avg ROC (AUC={metrics.get('roc_auc_micro', 0):.3f})")
        ax.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC data unavailable", ha="center", va="center")
    ax.set_title("ROC Curve"); ax.grid(True)

    # Precision-Recall curve
    prec, rec = metrics.get("precision_micro", np.array([])), metrics.get("recall_micro", np.array([]))
    ax        = axes[1]
    if prec.size:
        ax.plot(rec, prec, color="blue", lw=2,
                label=f"Micro-avg PR (AP={metrics.get('avg_precision_micro', 0):.3f})")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
    else:
        ax.text(0.5, 0.5, "PR data unavailable", ha="center", va="center")
    ax.set_title("Precision-Recall Curve"); ax.grid(True)

    plt.tight_layout()
    plt.savefig(config["eval_curves_plot_save_path"], bbox_inches="tight")
    plt.show()
    print(f"Evaluation curves saved -> {config['eval_curves_plot_save_path']}")

    # Confusion matrix
    cm          = metrics.get("confusion_matrix")
    class_names = config.get("class_names")
    if cm is None or class_names is None or len(class_names) != cm.shape[0]:
        print("Skipping confusion matrix — data unavailable or class mismatch.")
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n // 2), max(6, n // 2)))
    sns.heatmap(cm_norm, ax=ax, annot=(n <= 20), fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.set_title("Normalised Confusion Matrix")
    plt.xticks(rotation=60, ha="right", fontsize=max(6, 10 - n // 5))
    plt.yticks(rotation=0,  fontsize=max(6, 10 - n // 5))
    plt.tight_layout()
    plt.savefig(config["confusion_matrix_plot_save_path"], bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved -> {config['confusion_matrix_plot_save_path']}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_model_for_inference(model_path: str, config: Dict) -> Optional[FaceViT]:
    """
    Reconstruct FaceViT and load a saved state dict.
    Handles DataParallel 'module.' prefix automatically.

    Args:
        model_path: Path to the saved .pth state dict file.
        config:     Global CONFIG dict.

    Returns:
        Loaded FaceViT in eval mode, or None on failure.
    """
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        print("  File not found.")
        return None

    model = FaceViT(
        model_name    = config["vit_model_name"],
        pretrained    = False,
        embedding_dim = config["embedding_dim"],
        img_size      = config["img_size"],
    )

    state_dict = torch.load(model_path, map_location=config["device"])
    if all(k.startswith("module.") for k in state_dict):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

    model.load_state_dict(state_dict, strict=True)
    model.to(config["device"]).eval()
    print("  Model loaded successfully.")
    return model


def load_criterion_for_inference(
    criterion_path: str, config: Dict
) -> Optional[ArcFaceLoss]:
    """
    Reconstruct ArcFaceLoss and load a saved state dict.

    Args:
        criterion_path: Path to the saved criterion .pth file.
        config:         Global CONFIG dict.

    Returns:
        Loaded ArcFaceLoss in eval mode, or None on failure.
    """
    num_classes = config.get("num_classes")
    if num_classes is None:
        print("num_classes not set — cannot load criterion.")
        return None

    print(f"Loading criterion from: {criterion_path}")
    if not os.path.exists(criterion_path):
        print("  Criterion file not found. Evaluation metrics will be inaccurate.")
        return None

    criterion = ArcFaceLoss(
        in_features  = config["embedding_dim"],
        out_features = num_classes,
        s            = config["arcface_s"],
        m            = config["arcface_m"],
    )
    criterion.load_state_dict(
        torch.load(criterion_path, map_location=config["device"]), strict=True
    )
    criterion.to(config["device"]).eval()
    print("  Criterion loaded successfully.")
    return criterion


# ══════════════════════════════════════════════════════════════════════════════
# GALLERY & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def build_gallery(
    model: FaceViT,
    gallery_dir: str,
    transform: transforms.Compose,
    device: str,
    config: Dict,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build a gallery of mean L2-normalised embeddings per identity.

    Args:
        model:       Trained FaceViT in eval mode.
        gallery_dir: ImageFolder-structured directory of gallery images.
        transform:   Inference transform pipeline.
        device:      Torch device string.
        config:      Global CONFIG dict.

    Returns:
        Tuple of (gallery dict: name -> embedding, list of class names).
    """
    print(f"\nBuilding gallery from: {gallery_dir}")
    gallery: Dict[str, np.ndarray] = {}

    if not os.path.isdir(gallery_dir):
        print("  Gallery directory not found.")
        return gallery, []

    gallery_ds = ImageFolder(gallery_dir, transform=transform)
    if not gallery_ds.classes:
        print("  No identity sub-directories found.")
        return gallery, []

    loader = DataLoader(
        gallery_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
    )
    print(f"  Processing {len(gallery_ds)} images for {len(gallery_ds.classes)} identities...")

    all_embs, all_lbls = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            all_embs.append(model(images.to(device)).cpu().numpy())
            all_lbls.extend(labels.numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    all_lbls = np.array(all_lbls)

    for idx, name in enumerate(gallery_ds.classes):
        mask     = all_lbls == idx
        if not mask.any():
            continue
        mean_emb = all_embs[mask].mean(axis=0)
        norm     = np.linalg.norm(mean_emb)
        gallery[name] = mean_emb / norm if norm > 1e-6 else mean_emb

    print(f"  Gallery ready — {len(gallery)} identities.")
    return gallery, gallery_ds.classes


def recognize_faces_in_image(
    model: FaceViT,
    gallery: Dict[str, np.ndarray],
    transform: transforms.Compose,
    device: str,
    image_path: str,
    cascade_path: str,
    threshold: float,
    config: Dict,
) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """
    Detect faces in a static image and identify each against the gallery.

    Uses OpenCV Haar Cascade for detection and cosine similarity for
    matching against gallery embeddings.

    Args:
        model:        Trained FaceViT in eval mode.
        gallery:      Identity gallery from build_gallery.
        transform:    Inference transform pipeline.
        device:       Torch device string.
        image_path:   Path to the input image.
        cascade_path: Path to the Haar Cascade XML file.
        threshold:    Minimum cosine similarity to accept a match.
        config:       Global CONFIG dict.

    Returns:
        Tuple of (annotated BGR frame, list of result dicts with keys
        box, identity, score).
    """
    if not gallery or not os.path.exists(image_path):
        print("Gallery empty or image path invalid.")
        return None, []

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Failed to load Haar Cascade from {cascade_path}.")
        return None, []

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot read image: {image_path}")
        return None, []

    annotated    = frame.copy()
    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces        = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    print(f"Detected {len(faces)} face(s) in '{os.path.basename(image_path)}'")

    gallery_names  = list(gallery.keys())
    gallery_matrix = np.array(list(gallery.values()))
    results: List[Dict] = []

    model.eval()
    with torch.no_grad():
        for x, y, w, h in faces:
            roi    = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(roi)).unsqueeze(0).to(device)
            emb    = model(tensor).cpu().numpy()
            sims   = cosine_similarity(emb, gallery_matrix)[0]
            best_i = int(np.argmax(sims))
            score  = float(sims[best_i])

            if score >= threshold:
                identity = gallery_names[best_i]
                color    = (0, 255, 0)   # Green — recognised
            else:
                identity = "Unknown"
                color    = (0, 0, 255)   # Red — below threshold

            results.append({"box": (x, y, w, h), "identity": identity, "score": score})
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, f"{identity} ({score:.2f})",
                        (x, y - 10 if y > 20 else y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(f"Face Recognition — {os.path.basename(image_path)}")
    plt.axis("off")
    plt.show()

    return annotated, results


def infer_on_validation(
    model: FaceViT,
    gallery: Dict[str, np.ndarray],
    val_loader: DataLoader,
    device: str,
    threshold: float,
    config: Dict,
    num_images: int = 10,
) -> None:
    """
    Run gallery-based recognition on a sample of validation images.

    Uses preprocessed tensors directly (no Haar Cascade detection).
    Intended as a quick sanity-check after training.

    Args:
        model:       Trained FaceViT in eval mode.
        gallery:     Identity gallery from build_gallery.
        val_loader:  Validation DataLoader.
        device:      Torch device string.
        threshold:   Cosine similarity recognition threshold.
        config:      Global CONFIG (needs class_to_idx).
        num_images:  Maximum number of samples to evaluate.
    """
    if not gallery or val_loader is None:
        print("Gallery or validation loader unavailable.")
        return

    idx_to_class   = {v: k for k, v in config.get("class_to_idx", {}).items()}
    gallery_names  = list(gallery.keys())
    gallery_matrix = np.array(list(gallery.values()))

    print(f"\n── Validation Inference Sample (n={num_images}) ─────────────────")
    count = 0
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            if count >= num_images:
                break
            embs = model(images.to(device)).cpu().numpy()
            for i in range(images.size(0)):
                if count >= num_images:
                    break
                sims       = cosine_similarity(embs[i:i+1], gallery_matrix)[0]
                best_i     = int(np.argmax(sims))
                score      = float(sims[best_i])
                predicted  = gallery_names[best_i] if score >= threshold else "Unknown"
                true_label = idx_to_class.get(int(labels[i]), "?")
                match_icon = "v" if predicted == true_label else "x"
                print(f"  [{match_icon}] Sample {count+1:02d} | "
                      f"True={true_label:<20} Pred={predicted:<20} Score={score:.3f}")
                count += 1

    print(f"── Done ({count} samples) ──────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    End-to-end pipeline:
        1. Download dependencies
        2. Load data
        3. Train FaceViT with ArcFace loss
        4. Evaluate best checkpoint
        5. Build identity gallery and run sample inference
    """
    # Setup
    haar_available           = download_haar_cascade(CONFIG["haar_cascade_path"])
    train_loader, val_loader = get_dataloaders(CONFIG)
    if train_loader is None:
        raise SystemExit("Data loading failed. Check CONFIG['data_dir'].")

    # Initialise model, loss, optimiser, scheduler
    model = FaceViT(
        model_name    = CONFIG["vit_model_name"],
        pretrained    = CONFIG["pretrained"],
        embedding_dim = CONFIG["embedding_dim"],
        img_size      = CONFIG["img_size"],
    ).to(CONFIG["device"])

    criterion = ArcFaceLoss(
        in_features  = CONFIG["embedding_dim"],
        out_features = CONFIG["num_classes"],
        s            = CONFIG["arcface_s"],
        m            = CONFIG["arcface_m"],
    ).to(CONFIG["device"])

    optimizer = optim.AdamW(
        params       = [{"params": model.parameters()},
                        {"params": criterion.parameters()}],
        lr           = CONFIG["learning_rate"],
        weight_decay = CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )

    # Train
    history = train_model(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, CONFIG,
    )
    plot_training_history(history, CONFIG["history_plot_save_path"])

    # Evaluate best checkpoint
    inference_model = load_model_for_inference(CONFIG["model_save_path"], CONFIG)
    eval_criterion  = load_criterion_for_inference(CONFIG["criterion_save_path"], CONFIG)

    if inference_model and eval_criterion and val_loader:
        eval_metrics = evaluate_model(
            inference_model, eval_criterion, val_loader,
            CONFIG["device"], CONFIG["num_classes"], is_eval=False,
        )
        plot_evaluation_curves(eval_metrics, CONFIG)

    # Build gallery and run sample inference
    if inference_model:
        inf_transform = get_inference_transform(CONFIG["img_size"])
        gallery, _    = build_gallery(
            inference_model, CONFIG["gallery_dir"],
            inf_transform, CONFIG["device"], CONFIG,
        )

        if val_loader and gallery:
            infer_on_validation(
                inference_model, gallery, val_loader,
                CONFIG["device"], CONFIG["recognition_threshold"],
                CONFIG, num_images=10,
            )

        # Static image example — update path to a real image
        if CONFIG.get("class_names") and haar_available and gallery:
            first_class = CONFIG["class_names"][0]
            candidates  = [
                os.path.join(CONFIG["data_dir"], first_class, "1.jpg"),
                os.path.join(CONFIG["data_dir"], first_class, "image1.png"),
            ]
            test_path = next((p for p in candidates if os.path.exists(p)), None)
            if test_path:
                _, results = recognize_faces_in_image(
                    inference_model, gallery, inf_transform, CONFIG["device"],
                    test_path, CONFIG["haar_cascade_path"],
                    CONFIG["recognition_threshold"], CONFIG,
                )
                for r in results:
                    print(f"  Box={r['box']}  Identity={r['identity']}  Score={r['score']:.3f}")
            else:
                print("No example image found. Set test_path manually for static inference.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
