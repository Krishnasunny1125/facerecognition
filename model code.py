
!pip install timm opencv-python matplotlib scikit-learn seaborn -q

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
import timm
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import requests # For downloading haar cascade
from sklearn.metrics import (precision_recall_fscore_support, roc_curve, auc,
                             precision_recall_curve, average_precision_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from PIL import Image
import seaborn as sns # For plotting confusion matrix
import warnings
import sklearn # Import explicitly to check version

# Suppress specific warnings if needed (e.g., from sklearn)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')




CONFIG = {
    # === PATHS (MODIFY THESE) ===
    "data_dir": "/kaggle/input/recognition/DATASET IMAGES",  # <<< CHANGE THIS
    "gallery_dir": "/kaggle/input/recognition/DATASET IMAGES", # <<< CHANGE THIS (can be same as data_dir initially)
    "model_save_path": "/kaggle/working/face_vit_model_best.pth", # Output path
    "criterion_save_path": "/kaggle/working/face_vit_model_best_criterion.pth", # Output path for criterion weights
    "history_plot_save_path": "/kaggle/working/training_history.png",     # Output path
    "eval_curves_plot_save_path": "/kaggle/working/evaluation_curves.png", # Output path
    "confusion_matrix_plot_save_path": "/kaggle/working/confusion_matrix.png", # Output path
    "haar_cascade_path": '/kaggle/working/haarcascade_frontalface_default.xml', # Output path

    # === Model & Training ===
    "img_size": 224,
    "patch_size": 16,
    "embedding_dim": 512, # Dimension of the final face embedding
    "vit_model_name": 'vit_base_patch16_224', # ViT architecture from timm
    "pretrained": True,      # Use ImageNet pre-trained weights
    "learning_rate": 1e-4,   # Learning rate for fine-tuning
    "weight_decay": 0.01,    # Weight decay for AdamW optimizer
    "batch_size": 32,        # Adjust based on GPU memory (Kaggle T4: 16-32 often works)
    "num_epochs": 25,        # Number of training epochs
    "validation_split": 0.2, # Fraction of data for validation (e.g., 20%)
    "num_workers": 2,        # Number of data loading workers (Kaggle: 2 is usually good)

    # === Loss Function (ArcFace) ===
    "arcface_s": 30.0,       # ArcFace scale parameter
    "arcface_m": 0.50,       # ArcFace margin parameter (in radians)

    # === Inference ===
    "recognition_threshold": 0.60, # Cosine similarity threshold for recognition

    # === Internal - Auto-set ===
    "num_classes": None, # Will be set during data loading
    "class_names": None, # Will be set during data loading
    "class_to_idx": None,# Will be set during data loading
}

# Automatically detect device
CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {CONFIG['device']}")
print(f"Dataset path set to: {CONFIG['data_dir']}")
print(f"Model will be saved to: {CONFIG['model_save_path']}")

# Create working directory if it doesn't exist
os.makedirs("/kaggle/working/", exist_ok=True)




def download_haar_cascade(path):
    """Downloads the default frontal face Haar Cascade XML if not present."""
    if not os.path.exists(path):
        print(f"Downloading Haar Cascade file to {path}...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            response = requests.get(url, stream=True, timeout=30) # Added timeout
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Haar Cascade: {e}")
            print("Please ensure internet is enabled in the notebook settings and the URL is accessible.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            return False
    else:
        print("Haar Cascade file already exists.")
        return True

# Attempt to download
haar_cascade_available = download_haar_cascade(CONFIG["haar_cascade_path"])




def get_dataloaders(config):
    """Loads data, applies transforms, splits, and creates dataloaders."""
    img_size = config['img_size']
    batch_size = config['batch_size']
    validation_split = config['validation_split']
    num_workers = config['num_workers']
    data_dir = config['data_dir']

    # Define transformations
    # Normalization MUST match the ViT pre-training (ImageNet)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Added hue jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Validation/Inference transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Load the full dataset using ImageFolder
    try:
        # Check dataset structure first
        full_dataset_check = ImageFolder(data_dir)
        if not full_dataset_check.classes:
            print(f"Error: No subdirectories (classes) found in {data_dir}. Check dataset structure.")
            return None, None
        class_names = full_dataset_check.classes
        class_to_idx = full_dataset_check.class_to_idx
        num_classes = len(class_names)
        print(f"Found dataset with {len(full_dataset_check)} images in {num_classes} classes.")
        print(f"Class names: {class_names}")

        # Update config with dataset specifics
        config["num_classes"] = num_classes
        config["class_names"] = class_names
        config["class_to_idx"] = class_to_idx

        # Create datasets with appropriate transforms for splitting
        train_dataset_obj = ImageFolder(data_dir, transform=transform)
        val_dataset_obj = ImageFolder(data_dir, transform=val_transform)

    except FileNotFoundError:
        print(f"Error: Dataset directory not found at {data_dir}")
        print("Please ensure the path in CONFIG['data_dir'] is correct.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading dataset: {e}")
        return None, None

    # Split dataset into training and validation sets using indices
    total_len = len(train_dataset_obj)
    if total_len == 0:
        print("Error: Dataset is empty after loading.")
        return None, None

    indices = list(range(total_len))
    np.random.shuffle(indices) # Shuffle indices for random split
    val_len = int(np.floor(validation_split * total_len)) # Use floor to ensure integer
    train_indices, val_indices = indices[val_len:], indices[:val_len]

    # Ensure validation set is not empty if split is small
    if val_len == 0 and validation_split > 0 and total_len > 0:
        print("Warning: Validation split resulted in 0 samples. Adjust split or dataset size.")
        # Optionally, assign at least one sample to validation if possible
        # val_indices = train_indices[:1]
        # train_indices = train_indices[1:]
        # val_len = len(val_indices)
        # Or proceed with potentially empty validation loader

    # Create Subset objects
    train_subset = Subset(train_dataset_obj, train_indices)
    val_subset = Subset(val_dataset_obj, val_indices) # Use val_dataset_obj for validation subset

    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True) # drop_last can help stabilize training
    # Only create val_loader if val_subset has samples
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if len(val_subset) > 0 else None

    return train_loader, val_loader

# --- Load the data ---
train_loader, val_loader = get_dataloaders(CONFIG)

# Check if data loading was successful
if train_loader is None:
     print("\nData loading failed. Cannot proceed.")
     # Stop execution if data loading fails
     raise SystemExit("Could not load data. Please check CONFIG['data_dir'] and dataset structure.")
else:
     print("\nDataloaders created successfully.")
     if val_loader is None:
         print("Warning: Validation loader was not created (validation set might be empty).")



class FaceViT(nn.Module):
    """Vision Transformer model adapted for face embedding extraction."""
    def __init__(self, model_name, pretrained, embedding_dim, img_size):
        super().__init__()
        # Load the ViT backbone from timm, excluding the final classification layer (num_classes=0)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Get the feature dimension output by the ViT backbone
        try:
            vit_feature_dim = self.backbone.num_features
        except AttributeError:
            # Fallback: Infer feature dimension by passing a dummy input
            print("Could not directly access num_features. Inferring ViT feature dimension...")
            dummy_input = torch.randn(1, 3, img_size, img_size)
            with torch.no_grad():
                 dummy_output = self.backbone(dummy_input)
            vit_feature_dim = dummy_output.shape[-1]
            print(f"Inferred ViT feature dimension: {vit_feature_dim}")

        # Linear layer to project ViT features to the desired embedding dimension
        self.embedding_head = nn.Linear(vit_feature_dim, embedding_dim)

        # Initialization for the embedding head
        nn.init.xavier_uniform_(self.embedding_head.weight)
        if self.embedding_head.bias is not None:
            nn.init.constant_(self.embedding_head.bias, 0)

        print(f"Initialized FaceViT with backbone '{model_name}' (Pretrained: {pretrained})")
        print(f"ViT Feature Dim: {vit_feature_dim}, Output Embedding Dim: {embedding_dim}")

    def forward(self, x):
        # Pass input through the ViT backbone
        features = self.backbone(x)
        # Pass features through the embedding head
        embeddings = self.embedding_head(features)
        # L2 Normalize the embeddings - CRUCIAL for ArcFace and cosine similarity
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings




class ArcFaceLoss(nn.Module):
    """Implementation of the ArcFace loss function for face recognition."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features  # Dimension of input embeddings (e.g., 512)
        self.out_features = out_features # Number of classes (identities)
        self.s = s # Scale parameter (feature scaling factor)
        self.m = m # Angular margin penalty (in radians)

        # Learnable weight matrix (class centers/prototypes)
        # Shape: (num_classes, embedding_dimension)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) # Initialize weights

        self.easy_margin = easy_margin
        # Precompute constants for the margin calculation for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Threshold to prevent applying margin to angles near pi (prevents instability)
        # cos(theta + m) must be > cos(pi - m) = -cos(m)
        self.th = math.cos(math.pi - m)
        # Alternative penalty term (cosine - mm) for angles where cos(theta) < th
        self.mm = math.sin(math.pi - m) * m # sin(pi - m) = sin(m)

    def forward(self, embeddings, labels):
        # embeddings: (batch_size, embedding_dimension)
        # labels: (batch_size)
        # self.weight: (num_classes, embedding_dimension)

        # Calculate cosine similarity: cos(theta) = (embedding . weight) / (||embedding|| * ||weight||)
        # Since embeddings and weights are normalized, ||.|| = 1
        # Output shape: (batch_size, num_classes)
        cosine = F.linear(embeddings, F.normalize(self.weight)) # Use normalized weights

        # Calculate sin(theta) using sqrt(1 - cos^2(theta))
        # Clamp to avoid numerical issues with sqrt gradients near 1
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-6))

        # Calculate phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Ensure margin is only applied where cos(theta) > threshold 'th'
        # This helps prevent issues when the angle theta is already large (> pi - m)
        if self.easy_margin:
            # Simpler version: only apply margin if cos(theta) > 0
             phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: apply margin penalty only where cos(theta) > th
             phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding of labels for indexing
        # Shape: (batch_size, num_classes)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Construct the final logits:
        # Use 'phi' (cosine + margin) for the target class logit
        # Use 'cosine' (original cosine similarity) for non-target class logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits by the factor 's'
        output *= self.s

        # Calculate standard Cross-Entropy loss on the modified logits
        loss = F.cross_entropy(output, labels)
        return loss

    def get_logits(self, embeddings):
         """Calculates logits (scaled cosine similarities) for evaluation/inference."""
         # No gradients needed here
         with torch.no_grad():
             # Ensure weights are normalized for cosine similarity calculation
             normalized_weight = F.normalize(self.weight)
             # Calculate cosine similarity
             cosine = F.linear(embeddings, normalized_weight)
         # Return scaled cosine similarity (logits)
         return cosine * self.s

print("ArcFaceLoss class defined.")



def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, config):
    """Trains the model and saves the best version based on validation metric."""
    device = config['device']
    num_epochs = config['num_epochs']
    model_save_path = config['model_save_path']
    criterion_save_path = config['criterion_save_path']
    num_classes = config['num_classes'] # Get num_classes from config

    print(f"--- Starting Training ---")
    print(f"Device: {device}, Epochs: {num_epochs}, Batch Size: {config['batch_size']}, LR: {config['learning_rate']}")
    print(f"Saving best model to: {model_save_path}")
    print(f"Saving best criterion state to: {criterion_save_path}")

    # Choose metric to track for saving best model (e.g., accuracy or AUC)
    best_val_metric = 0.0
    best_epoch = -1
    metric_to_track = 'accuracy' # or 'roc_auc_micro'

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        criterion.train() # Set criterion to train mode
        running_train_loss = 0.0
        num_train_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            batch_s = images.size(0)
            num_train_samples += batch_s

            optimizer.zero_grad()
            embeddings = model(images)
            loss = criterion(embeddings, labels) # Use ArcFace loss for training
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * batch_s

        epoch_train_loss = running_train_loss / num_train_samples if num_train_samples > 0 else 0
        history['train_loss'].append(epoch_train_loss)

        # --- Validation Phase ---
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        epoch_val_auc = 0.0
        val_metrics = {} # Initialize val_metrics dict
        if val_loader:
            # Call updated evaluate_model for detailed metrics
            val_metrics = evaluate_model(model, criterion, val_loader, device, num_classes, is_eval=True)
            epoch_val_loss = val_metrics.get('loss', 0.0)
            epoch_val_acc = val_metrics.get('accuracy', 0.0)
            epoch_val_auc = val_metrics.get('roc_auc_micro', 0.0) # Get micro-AUC

            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            history['val_auc'].append(epoch_val_auc)
        else:
            # Handle case with no validation loader
             history['val_loss'].append(0.0)
             history['val_acc'].append(0.0)
             history['val_auc'].append(0.0)


        # --- Learning Rate Scheduler Step ---
        if scheduler:
            scheduler.step() # Step the scheduler after each epoch
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config['learning_rate']

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # --- Epoch Summary ---
        print(f'Epoch [{epoch+1:02d}/{num_epochs}] ({epoch_duration:.2f}s) -> '
              f'LR: {current_lr:.6f} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {epoch_val_acc:.4f} | '
              f'Val AUC: {epoch_val_auc:.4f}')

        # --- Save Best Model & Criterion ---
        # Use .get() for safer access to potentially missing keys if val_loader was None
        current_metric_to_save = val_metrics.get(metric_to_track, 0.0) if val_loader else 0.0

        if current_metric_to_save > best_val_metric:
            best_val_metric = current_metric_to_save
            best_epoch = epoch + 1
            try:
                torch.save(model.state_dict(), model_save_path)
                # Save the criterion state dict (contains learned weights)
                torch.save(criterion.state_dict(), criterion_save_path)
                print(f"    ----> New best {metric_to_track}: {best_val_metric:.4f}. Model & Criterion saved.")
            except Exception as e:
                 print(f"    ----> Error saving model/criterion state: {e}")

    print("\n--- Training Finished ---")
    if best_epoch != -1:
        print(f"Best Validation {metric_to_track.capitalize()} achieved: {best_val_metric:.4f} at Epoch {best_epoch}")
    else:
        print("No best model saved (validation metric did not improve or no validation set).")
    return history


# --- EVALUATION FUNCTION (WITH DETAILED METRICS including Overall FPR) ---
def evaluate_model(model, criterion, dataloader, device, num_classes, is_eval=False):
    """Evaluates the model, calculates detailed metrics including Overall FPR, and returns them."""
    model.eval() # Set model to evaluation mode
    criterion.eval() # Set criterion to evaluation mode

    all_labels = []
    all_predictions = [] # Store predicted class indices
    all_scores = []      # Store raw scores/logits for all classes

    running_loss = 0.0
    total_samples = 0

    # Default metrics dictionary in case of issues
    default_metrics = {'loss': 0.0, 'accuracy': 0.0, 'roc_auc_micro': 0.0, 'avg_precision_micro': 0.0,
                       'mAP_macro': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0,
                       'f1_weighted': 0.0, 'fpr_micro': np.array([]), 'tpr_micro': np.array([]),
                       'precision_micro': np.array([]), 'recall_micro': np.array([]),
                       'overall_fpr': 0.0, # Add placeholder for overall FPR
                       'confusion_matrix': None}
    metrics = default_metrics.copy()

    if dataloader is None:
        print("Warning: Dataloader is None in evaluate_model. Returning default metrics.")
        return metrics
    if num_classes is None or num_classes <= 0:
        print(f"Warning: Invalid num_classes ({num_classes}) in evaluate_model. Returning default metrics.")
        return metrics


    with torch.no_grad(): # Disable gradient calculations for evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            logits = criterion.get_logits(embeddings)
            loss = F.cross_entropy(logits, labels)
            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_scores.append(logits.cpu().numpy())
            _, predicted_indices = torch.max(logits, 1)
            all_predictions.extend(predicted_indices.cpu().numpy())
            total_samples += labels.size(0)

    if total_samples == 0:
         print("Warning: No samples processed during evaluation. Returning default metrics.")
         return metrics # Return default if no samples processed

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.concatenate(all_scores, axis=0)

    # --- Calculate Metrics ---
    try:
        metrics['loss'] = running_loss / total_samples
        metrics['accuracy'] = np.mean(all_predictions == all_labels) # More robust way
        labels_binarized = label_binarize(all_labels, classes=range(num_classes))

        if num_classes == 1:
            print("Warning: Only one class found. Skipping multi-class metrics (ROC, PR, mAP, FPR).")
            metrics.update({k: v for k, v in default_metrics.items() if k not in ['loss', 'accuracy', 'confusion_matrix']})
            try: # Still try to get CM for single class
                metrics['confusion_matrix'] = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
            except Exception as e_cm:
                 print(f"Could not calculate confusion matrix for single class: {e_cm}")
        else: # Multi-class case
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            metrics['precision_weighted'] = precision
            metrics['recall_weighted'] = recall
            metrics['f1_weighted'] = f1

            fpr_micro, tpr_micro, _ = roc_curve(labels_binarized.ravel(), all_scores.ravel())
            metrics['roc_auc_micro'] = auc(fpr_micro, tpr_micro) if len(np.unique(labels_binarized.ravel())) > 1 else 0.0 # Check if multiple classes present after ravel
            metrics['fpr_micro'] = fpr_micro
            metrics['tpr_micro'] = tpr_micro

            precision_micro, recall_micro, _ = precision_recall_curve(labels_binarized.ravel(), all_scores.ravel())
            metrics['avg_precision_micro'] = average_precision_score(labels_binarized, all_scores, average='micro') if len(np.unique(labels_binarized.ravel())) > 1 else 0.0
            metrics['precision_micro'] = precision_micro
            metrics['recall_micro'] = recall_micro

            try:
                 metrics['mAP_macro'] = average_precision_score(labels_binarized, all_scores, average='macro')
            except ValueError as e:
                 print(f"Warning: Could not compute macro mAP. Check class distribution. Error: {e}")
                 metrics['mAP_macro'] = 0.0

            # --- Calculate Confusion Matrix and Overall FPR ---
            try:
                cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
                metrics['confusion_matrix'] = cm

                # Calculate Overall FPR from Confusion Matrix
                FP = cm.sum(axis=0) - np.diag(cm) # Sum of each column (predicted class) minus the diagonal (TP)
                FN = cm.sum(axis=1) - np.diag(cm) # Sum of each row (true class) minus the diagonal (TP)
                TP = np.diag(cm)
                # TN for class i = total samples - TP_i - FP_i - FN_i
                # Sum TN across all classes: Sum(total_samples - TP_i - FP_i - FN_i)
                # = (total_samples * num_classes) - Sum(TP) - Sum(FP) - Sum(FN)
                # Simpler: TN = cm.sum() - (FP + FN + TP)
                TN = cm.sum() - (FP + FN + TP) # Sum of all elements minus FP, FN, TP for each class

                # Sum FP and TN across all classes
                total_FP = FP.sum()
                total_TN = TN.sum()

                # Calculate Overall FPR = Total FP / (Total FP + Total TN)
                overall_fpr = total_FP / (total_FP + total_TN) if (total_FP + total_TN) > 0 else 0.0
                metrics['overall_fpr'] = overall_fpr # Store the calculated overall FPR

            except Exception as e_cm:
                 print(f"Could not calculate confusion matrix or Overall FPR: {e_cm}")
                 metrics['confusion_matrix'] = None
                 metrics['overall_fpr'] = 0.0 # Set default if calculation fails


    except Exception as e_metrics:
        print(f"An error occurred during metric calculation: {e_metrics}")
        print("Returning default metrics.")
        return default_metrics # Return default if any major error occurs


    # --- Print Detailed Results (only if not called during training validation steps) ---
    if not is_eval:
         print(f'\n--- Evaluation Results ---')
         print(f'Dataset size: {total_samples} samples, Classes: {num_classes}')
         print(f"Loss: {metrics.get('loss', 0.0):.4f}")
         print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
         print(f"Overall FPR: {metrics.get('overall_fpr', 0.0):.4f}") # <-- Print Overall FPR
         print(f"Weighted Precision: {metrics.get('precision_weighted', 0.0):.4f}")
         print(f"Weighted Recall: {metrics.get('recall_weighted', 0.0):.4f}")
         print(f"Weighted F1-Score: {metrics.get('f1_weighted', 0.0):.4f}")
         print(f"Micro-AUC (ROC): {metrics.get('roc_auc_micro', 0.0):.4f}")
         print(f"Micro-AP (PR): {metrics.get('avg_precision_micro', 0.0):.4f}")
         print(f"Macro-mAP: {metrics.get('mAP_macro', 0.0):.4f}")

    return metrics

print("Training and Evaluation functions (Cell 7) defined.")





def plot_history(history, save_path):
    """Plots training/validation loss, accuracy, and AUC curves."""
    if not history or not history.get('train_loss'):
        print("History object is empty or missing data. Cannot plot training history.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    num_plots = 3 # Loss, Accuracy, AUC
    plt.figure(figsize=(8, 4 * num_plots)) # Adjust figure size

    # Plot Loss
    plt.subplot(num_plots, 1, 1)
    plt.plot(epochs, history.get('train_loss', [0]*len(epochs)), 'bo-', label='Training Loss')
    plt.plot(epochs, history.get('val_loss', [0]*len(epochs)), 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs') # Label only bottom plot
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_ticklabels([]) # Remove x-axis labels for top plots

    # Plot Accuracy
    plt.subplot(num_plots, 1, 2)
    plt.plot(epochs, history.get('val_acc', [0]*len(epochs)), 'go-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    # plt.xlabel('Epochs') # Label only bottom plot
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05) # Set y-axis limits for accuracy
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_ticklabels([]) # Remove x-axis labels

    # Plot AUC
    plt.subplot(num_plots, 1, 3)
    plt.plot(epochs, history.get('val_auc', [0]*len(epochs)), 'mo-', label='Validation Micro-AUC')
    plt.title('Validation Micro-AUC (ROC)')
    plt.xlabel('Epochs') # Label only bottom plot
    plt.ylabel('AUC')
    plt.ylim(0, 1.05) # Set y-axis limits for AUC
    plt.legend()
    plt.grid(True)

    plt.tight_layout(pad=2.0) # Add padding between subplots
    try:
        plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches='tight'
        print(f"Training history plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    plt.show() # Display the plot


# --- PLOTTING FUNCTION FOR EVALUATION CURVES ---
def plot_evaluation_curves(metrics, config):
    """Plots ROC, Precision-Recall curves, and Confusion Matrix based on metrics data."""
    print("\n--- Generating Evaluation Plots ---")

    save_curves_path = config['eval_curves_plot_save_path']
    save_cm_path = config['confusion_matrix_plot_save_path']

    plt.figure(figsize=(14, 6)) # Figure for ROC and PR

    # --- Plot ROC Curve ---
    plt.subplot(1, 2, 1)
    fpr = metrics.get('fpr_micro', np.array([]))
    tpr = metrics.get('tpr_micro', np.array([]))
    roc_auc = metrics.get('roc_auc_micro', 0.0)

    if fpr.size > 0 and tpr.size > 0:
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc:0.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line for reference
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'ROC data not available\n(Requires >1 class)', horizontalalignment='center', verticalalignment='center', fontsize=10)
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.grid(True)
        print("ROC curve data missing or invalid (possibly only one class).")


    # --- Plot Precision-Recall Curve ---
    plt.subplot(1, 2, 2)
    recall = metrics.get('recall_micro', np.array([]))
    precision = metrics.get('precision_micro', np.array([]))
    avg_precision = metrics.get('avg_precision_micro', 0.0)

    if precision.size > 0 and recall.size > 0:
        # Plot Precision vs Recall
        # Note: precision_recall_curve outputs recall sorted ascendingly.
        # Plotting P vs R is standard.
        plt.plot(recall, precision, color='blue', lw=2, label=f'Micro-average PR (AP = {avg_precision:0.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left") # Often best location for PR curves
        plt.grid(True)
    else:
         plt.text(0.5, 0.5, 'PR data not available\n(Requires >1 class)', horizontalalignment='center', verticalalignment='center', fontsize=10)
         plt.title('Precision-Recall Curve')
         plt.grid(True)
         print("Precision-Recall curve data missing or invalid (possibly only one class).")

    plt.tight_layout()
    # Save the ROC/PR plot
    try:
        plt.savefig(save_curves_path, bbox_inches='tight')
        print(f"Evaluation curves plot saved to {save_curves_path}")
    except Exception as e:
        print(f"Error saving evaluation curves plot: {e}")
    plt.show() # Display the plot


    # --- Plot Confusion Matrix ---
    cm = metrics.get('confusion_matrix', None)
    class_names = config.get('class_names', None) # Get class names from config

    if cm is not None and class_names is not None and len(class_names) == cm.shape[0]:
        plt.figure(figsize=(max(8, len(class_names)//2), max(6, len(class_names)//2.5))) # Adjust size

        # Normalize CM row-wise (shows % of true labels predicted as each class)
        with np.errstate(divide='ignore', invalid='ignore'): # Ignore potential division by zero if a class has no samples
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized) # Replace NaNs (from 0/0) with 0

        sns.heatmap(cm_normalized, annot=False, fmt=".2f", cmap="Blues", # annot=True can be cluttered
                    xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1) # Ensure color scale is 0-1
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        plt.xticks(rotation=60, ha='right', fontsize=max(6, 10 - len(class_names)//5)) # Adjust font size
        plt.yticks(rotation=0, fontsize=max(6, 10 - len(class_names)//5))
        plt.tight_layout()
        try:
             plt.savefig(save_cm_path, bbox_inches='tight')
             print(f"Confusion matrix plot saved to {save_cm_path}")
        except Exception as e:
             print(f"Error saving confusion matrix plot: {e}")
        plt.show() # Display the plot
    elif cm is not None:
         print("\nCannot plot confusion matrix: Class names mismatch or not found in CONFIG.")
         print(f"CM shape: {cm.shape}, Num classes in config: {len(class_names) if class_names else 'None'}")
    else:
         print("\nConfusion matrix data not available for plotting.")

print("Visualization functions defined.")








def load_model_for_inference(model_path, config):
    """Loads the trained FaceViT model structure and weights."""
    print(f"\n--- Initializing model structure for loading ---")
    try:
        model = FaceViT(
            model_name=config["vit_model_name"],
            pretrained=False, # Always False when loading state_dict from your trained model
            embedding_dim=config["embedding_dim"],
            img_size=config["img_size"]
        )
    except Exception as e:
        print(f"Error initializing model structure: {e}")
        print("Ensure CONFIG values (vit_model_name, embedding_dim, img_size) are correct.")
        return None

    print(f"--- Loading model state from: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        # Load state dict, handling potential DataParallel wrapper and mapping to current device
        state_dict = torch.load(model_path, map_location=torch.device(config['device']))

        # Create a new state_dict if 'module.' prefix is found (from DataParallel/DDP)
        if isinstance(state_dict, OrderedDict) and all(k.startswith('module.') for k in state_dict.keys()):
            print("Detected model trained with DataParallel/DDP. Removing 'module.' prefix for loading.")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[len('module.'):] # remove `module.` prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
        else:
             # Load directly if no 'module.' prefix
             model.load_state_dict(state_dict, strict=True)

        # Move the entire model to the target device *after* loading weights
        model.to(config['device'])
        model.eval() # Set to evaluation mode (crucial!)
        print(f"Model state loaded successfully to {config['device']}.")
        return model

    except RuntimeError as e:
         print(f"Error loading model state_dict (possible architecture mismatch): {e}")
         print("Ensure the current model definition matches the saved checkpoint.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred loading model state_dict: {e}")
        return None

# --- FUNCTION TO LOAD CRITERION ---
def load_criterion_for_inference(criterion_path, config):
    """Loads the trained ArcFace criterion weights (needed for evaluation)."""
    # Requires num_classes to be set in config
    num_classes = config.get('num_classes')
    if num_classes is None:
         print("Error: num_classes not set in CONFIG. Cannot initialize criterion.")
         return None

    print(f"\n--- Initializing criterion structure for loading ---")
    try:
        criterion = ArcFaceLoss(
            in_features=config["embedding_dim"],
            out_features=num_classes,
            s=config["arcface_s"],
            m=config["arcface_m"]
        )
    except Exception as e:
         print(f"Error initializing criterion structure: {e}")
         return None

    print(f"--- Loading criterion state from: {criterion_path} ---")
    if not os.path.exists(criterion_path):
        print(f"Warning: Criterion state file not found at {criterion_path}.")
        print("Evaluation metrics like accuracy will be incorrect without these weights.")
        return None # Cannot evaluate properly without the learned weights

    try:
        # Load the criterion state dict, mapping location to the correct device
        criterion_state_dict = torch.load(criterion_path, map_location=config["device"])
        criterion.load_state_dict(criterion_state_dict, strict=True)
        criterion.to(config['device'])
        criterion.eval() # Set criterion to evaluation mode
        print(f"Criterion state loaded successfully to {config['device']}.")
        return criterion

    except RuntimeError as e:
         print(f"Error loading criterion state_dict (possible mismatch): {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred loading criterion state_dict: {e}.")
        return None

# --- INFERENCE TRANSFORM ---
def get_inference_transform(img_size):
     """Returns the transformation pipeline for inference (no augmentation)."""
     return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Use ImageNet stats
    ])

# --- BUILD GALLERY ---
def build_gallery(model, gallery_image_dir, transform, device, config):
    """Builds a gallery of average L2-normalized embeddings for each identity."""
    gallery = {} # {identity_name: avg_normalized_embedding}
    gallery_class_names = [] # Store names specific to gallery folder structure

    if not os.path.isdir(gallery_image_dir):
         print(f"Error: Gallery image directory '{gallery_image_dir}' not found.")
         return gallery, gallery_class_names

    print(f"\n--- Building face embedding gallery from: {gallery_image_dir} ---")
    try:
        # Use a separate ImageFolder instance for the gallery
        gallery_dataset = ImageFolder(gallery_image_dir, transform=transform)
        if not gallery_dataset.classes:
            print(f"Warning: No subdirectories (classes/persons) found in '{gallery_image_dir}'. Gallery will be empty.")
            return {}, []
        gallery_class_names = gallery_dataset.classes # Names specific to gallery folders

        # Use a DataLoader to process gallery images efficiently
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=config['batch_size'],
            shuffle=False, # No need to shuffle gallery
            num_workers=config['num_workers']
        )
        print(f"Processing {len(gallery_dataset)} images for {len(gallery_class_names)} identities in gallery...")

        embeddings_list = []
        labels_list = [] # Labels correspond to gallery folder indices (0, 1, 2...)
        model.eval()
        with torch.no_grad():
            for images, labels in gallery_loader:
                images = images.to(device)
                # Model outputs L2-normalized embeddings directly
                batch_embeddings = model(images).cpu().numpy()
                embeddings_list.append(batch_embeddings)
                labels_list.extend(labels.numpy())

        if not embeddings_list:
            print("Warning: No embeddings generated from gallery images.")
            return {}, gallery_class_names

        all_embeddings = np.concatenate(embeddings_list, axis=0)
        all_labels = np.array(labels_list)

        # Calculate the average embedding for each identity
        for i, class_name in enumerate(gallery_class_names):
             identity_indices = np.where(all_labels == i)[0]
             if len(identity_indices) > 0:
                 identity_embeddings = all_embeddings[identity_indices]
                 # Average the normalized embeddings
                 avg_embedding = np.mean(identity_embeddings, axis=0)
                 # Re-normalize the average embedding to ensure it has unit length
                 norm = np.linalg.norm(avg_embedding)
                 if norm > 1e-6: # Avoid division by zero
                     avg_embedding_normalized = avg_embedding / norm
                 else:
                      print(f"Warning: Zero norm for average embedding of '{class_name}'. Using unnormalized.")
                      avg_embedding_normalized = avg_embedding # Use as is if norm is zero

                 gallery[class_name] = avg_embedding_normalized # Store the normalized average
                 # print(f" - Added '{class_name}' (avg over {len(identity_indices)} images) to gallery.")
             else:
                 # This case shouldn't usually happen with ImageFolder structure if folders exist and contain images
                 print(f"Warning: No images found or processed for gallery class index {i} ('{class_name}').")

    except FileNotFoundError:
         print(f"Error: Problem accessing gallery directory '{gallery_image_dir}' or its contents.")
         return {}, []
    except Exception as e:
         print(f"An unexpected error occurred building gallery: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback for debugging
         return {}, gallery_class_names

    if not gallery:
         print("Warning: Gallery is empty after processing.")
    else:
         print(f"Gallery built successfully with {len(gallery)} identities.")
    return gallery, gallery_class_names

print("Inference helper functions defined.")





def recognize_faces_in_image(model, gallery, transform, device, image_path, cascade_path, threshold, config):
    """Detects faces, extracts embeddings, compares to gallery, and draws results."""
    # --- Input Validation ---
    if not gallery:
        print("Error: Cannot perform recognition - Gallery is empty.")
        return None, None # Return None for frame and results
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        return None, None
    if not haar_cascade_available or not os.path.exists(cascade_path):
         print("Error: Haar cascade file missing or unavailable.")
         return None, None

    # --- Load Detector ---
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error loading Haar Cascade from {cascade_path}")
        return None, None

    # --- Load Image ---
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    frame_display = frame.copy() # Create a copy for drawing annotations

    # --- Face Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adjust detection parameters if needed
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    print(f"Detected {len(faces)} potential face(s) in '{os.path.basename(image_path)}'")

    # --- Prepare Gallery Data ---
    if not gallery: # Double check after potential build errors
        print("Error: Gallery is empty (post-check).")
        return frame_display, [] # Return frame even if no gallery

    gallery_names = list(gallery.keys())
    # Ensure gallery embeddings are in a NumPy array (N_identities, embedding_dim)
    gallery_embeddings = np.array(list(gallery.values()))

    results = []
    model.eval() # Ensure model is in eval mode

    # --- Process Each Detected Face ---
    with torch.no_grad():
        for i, (x, y, w, h) in enumerate(faces):
            # Extract Face ROI (Region of Interest)
            face_roi_bgr = frame[y:y+h, x:x+w]
            # Default values
            identity = "Processing Error"
            best_score = -1.0 # Use -1 as default for cosine similarity
            color = (0, 255, 255) # Yellow

            try:
                if face_roi_bgr.size == 0: # Check if ROI is empty
                    print(f"Warning: Empty face ROI detected for face #{i+1}. Skipping.")
                    identity = "Empty ROI"
                    continue # Skip to next face

                # Preprocess the ROI
                face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_roi_rgb)
                input_tensor = transform(pil_image).unsqueeze(0).to(device) # Add batch dimension

                # Get L2-Normalized Embedding from Model
                embedding = model(input_tensor).cpu().numpy() # Shape: (1, embedding_dim)

                # Compare embedding with gallery using Cosine Similarity
                # Result shape: (1, num_gallery_identities)
                similarities = cosine_similarity(embedding, gallery_embeddings)[0]

                # Find the best match
                best_match_idx = np.argmax(similarities)
                best_score = similarities[best_match_idx]

                # Recognize based on threshold
                if best_score >= threshold:
                    identity = gallery_names[best_match_idx]
                    color = (0, 255, 0) # Green for recognized match
                else:
                    identity = "Unknown"
                    color = (0, 0, 255) # Red for unknown (below threshold)

            except Exception as e:
                print(f"Error processing face #{i+1} at box ({x},{y},{w},{h}): {e}")
                # Keep identity="Processing Error", color=Yellow

            # Store result
            results.append({
                "box": (x, y, w, h),
                "identity": identity,
                "score": float(best_score) # Ensure score is standard float
            })

            # --- Draw Annotation on the Display Frame ---
            display_text = f"{identity} ({best_score:.2f})"
            # Draw bounding box
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 2)
            # Put text slightly above the box
            text_y = y - 10 if y - 10 > 10 else y + 10 # Adjust text position if box is near top
            cv2.putText(frame_display, display_text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Display Final Image ---
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
    plt.title(f"Face Recognition Results: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

    return frame_display, results

print("Static image inference function defined.")



def infer_on_validation(model, gallery, val_loader, device, threshold, config, num_images=10):
    """Runs inference on a few images from the validation set (tensor-based comparison)."""
    # --- Input Checks ---
    if not gallery:
        print("Gallery empty, cannot run validation inference.")
        return
    if val_loader is None:
        print("Validation loader not available, skipping validation inference.")
        return
    idx_to_class_map = {v: k for k, v in config.get('class_to_idx', {}).items()}
    if not idx_to_class_map:
        print("Class-to-index mapping not found in config. Cannot map results.")
        return

    print(f"\n--- Running Inference on first ~{num_images} validation samples (Tensor-based) ---")
    print("NOTE: Compares whole image embedding from DataLoader against gallery.")
    print("      Haar cascade detection is SKIPPED in this specific function.")

    model.eval() # Ensure model is in evaluation mode
    count = 0

    # --- Prepare Gallery Data ---
    gallery_names = list(gallery.keys())
    gallery_embeddings = np.array(list(gallery.values()))

    # --- Process Batches from Validation Loader ---
    with torch.no_grad():
        for images, labels in val_loader:
            if count >= num_images: break # Stop after processing desired number

            images = images.to(device)
            labels = labels.cpu().numpy() # Get true labels for comparison

            # Get L2-Normalized Embeddings from Model
            embeddings_batch = model(images).cpu().numpy() # Shape (batch, dim)

            # --- Process Each Image in the Batch ---
            for i in range(images.size(0)):
                if count >= num_images: break

                embedding = embeddings_batch[i:i+1] # Keep shape (1, dim) for cosine_similarity
                true_label_idx = labels[i]
                true_label_name = idx_to_class_map.get(true_label_idx, f"INVALID_IDX_{true_label_idx}")

                # Compare embedding to gallery
                similarities = cosine_similarity(embedding, gallery_embeddings)[0]
                best_match_idx = np.argmax(similarities)
                best_score = similarities[best_match_idx]

                # Determine predicted identity based on threshold
                predicted_identity = "Unknown"
                if best_score >= threshold:
                     predicted_identity = gallery_names[best_match_idx]

                # Print comparison result
                print(f"Sample {count+1:02d}: True='{true_label_name}', Predicted='{predicted_identity}' (Score: {best_score:.3f})")
                count += 1

    print(f"--- Finished validation inference sample ({count} images processed) ---")

print("Validation set inference function defined.")



if train_loader is None or CONFIG.get('num_classes') is None:
    raise SystemExit("Dataloaders or num_classes not initialized. Check Cell 4 execution and dataset.")
if val_loader is None:
    print("Warning: Validation loader is None. Training will proceed without validation steps.")

print("\n--- Initializing Model, Loss, Optimizer, and Scheduler ---")

# 1. Model
try:
    model = FaceViT(
        model_name=CONFIG["vit_model_name"],
        pretrained=CONFIG["pretrained"],
        embedding_dim=CONFIG["embedding_dim"],
        img_size=CONFIG["img_size"]
    ).to(CONFIG["device"])
except Exception as e:
     print(f"FATAL: Failed to initialize model: {e}")
     raise SystemExit("Model initialization failed.")


# 2. Loss Function
try:
    criterion = ArcFaceLoss(
        in_features=CONFIG["embedding_dim"],
        out_features=CONFIG["num_classes"], # Use num_classes obtained from data
        s=CONFIG["arcface_s"],
        m=CONFIG["arcface_m"]
    ).to(CONFIG["device"])
except Exception as e:
    print(f"FATAL: Failed to initialize ArcFaceLoss: {e}")
    raise SystemExit("Loss function initialization failed.")


# 3. Optimizer (AdamW recommended for Transformers)
try:
    optimizer = optim.AdamW(
        params=[
            {'params': model.parameters()},
            {'params': criterion.parameters()} # Ensure ArcFace weights are optimized
        ],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
except Exception as e:
    print(f"FATAL: Failed to initialize Optimizer: {e}")
    raise SystemExit("Optimizer initialization failed.")


# 4. Learning Rate Scheduler (Cosine Annealing is a good choice)
try:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"], # Number of epochs for one cycle
        eta_min=1e-6 # Minimum learning rate
    )
except Exception as e:
    print(f"FATAL: Failed to initialize Scheduler: {e}")
    raise SystemExit("Scheduler initialization failed.")


print("\nModel, Criterion, Optimizer, Scheduler Initialized Successfully.")




train_enabled = True # Set to False to skip training

history = None # Initialize history dictionary

if train_enabled:
    if train_loader is None:
         print("Cannot train: Training dataloader is not available.")
    else:
         print("\n=== Starting Model Training ===")
         training_start_time = time.time()
         history = train_model(
             model, criterion, optimizer, scheduler, train_loader, val_loader, CONFIG
         )
         training_end_time = time.time()
         print(f"\n=== Training Complete (Duration: {(training_end_time - training_start_time)/60:.2f} minutes) ===")

         # --- Plotting Training History ---
         if history:
              print("\n--- Plotting Training History ---")
              plot_history(history, CONFIG["history_plot_save_path"])
         else:
              print("\nNo training history recorded.")

else:
    print("\n--- Training Skipped (train_enabled = False) ---")
    print(f"Will attempt to load model from {CONFIG['model_save_path']} in the next cell.")





print("\n=== Loading Model and Criterion for Evaluation ===")

# Use the helper functions to load state dicts
inference_model = load_model_for_inference(CONFIG["model_save_path"], CONFIG)
eval_criterion = load_criterion_for_inference(CONFIG["criterion_save_path"], CONFIG)


# --- Run Final Evaluation ---
evaluation_metrics = None # Initialize to None

if inference_model is not None and eval_criterion is not None:
    if val_loader is not None:
        print("\n--- Evaluating Best Loaded Model on Validation Set ---")
        eval_start_time = time.time()
        evaluation_metrics = evaluate_model(
            inference_model,
            eval_criterion,
            val_loader,
            CONFIG["device"],
            CONFIG["num_classes"], # Pass num_classes from config
            is_eval=False # Set is_eval=False to print detailed metrics summary
        )
        eval_end_time = time.time()
        print(f"--- Evaluation Complete (Duration: {eval_end_time - eval_start_time:.2f} seconds) ---")


        # --- Plot Evaluation Curves ---
        if evaluation_metrics:
             # Plot ROC, PR, and optionally Confusion Matrix
             plot_evaluation_curves(evaluation_metrics, CONFIG)
        else:
             print("\nEvaluation did not return metrics. Skipping curve plotting.")

    else:
        print("\nValidation loader ('val_loader') is None. Skipping final evaluation.")

elif inference_model is not None and eval_criterion is None:
     print("\nSkipping evaluation because the required criterion weights could not be loaded.")
     print("Accurate evaluation metrics cannot be calculated without the learned ArcFace weights.")

else:
    print("\nSkipping evaluation because the model failed to load.")

print("\n=== Evaluation Phase Finished ===")



# %% [markdown]
# ## Cell 15: Main Execution - Gallery Building and Inference Examples

print("\n=== Gallery Building and Inference Examples ===")

if inference_model is None:
     print("\nCannot proceed with inference: Model not loaded or failed to load.")
else:
    # --- Build Gallery ---
    # Note: Ensure gallery_dir points to a directory structured like the training data
    # (subfolders named after identities) containing representative images.
    inference_transform = get_inference_transform(CONFIG['img_size'])
    gallery, gallery_class_names = build_gallery(
        inference_model,
        CONFIG["gallery_dir"],
        inference_transform,
        CONFIG["device"],
        CONFIG # Pass full config
    )

    # --- Example: Inference on a Single Static Image ---
    # <<< --- CHANGE this path to a valid test image accessible by the notebook --- >>>
    # Example: Find an image within your input dataset if testing in Kaggle
    if CONFIG.get("class_names") and len(CONFIG["class_names"]) > 0:
         # Construct a plausible test path using the first class and trying a common name
         first_class_name = CONFIG["class_names"][0]
         first_class_name1 = CONFIG["class_names"][1]
         # Try a few common image extensions/names
         potential_paths = [
             os.path.join(CONFIG["data_dir"], first_class_name, "1.jpg"),
             os.path.join(CONFIG["data_dir"], first_class_name1, "10.jpg"),
             os.path.join(CONFIG["data_dir"], first_class_name, "image1.png"),
             os.path.join(CONFIG["data_dir"], first_class_name, f"{first_class_name}_001.jpg"),
         ]
         test_image_path = None
         for p_path in potential_paths:
             if os.path.exists(p_path):
                 test_image_path = p_path
                 print(f"Auto-detected test image: {test_image_path}")
                 break
         if test_image_path is None:
              print(f"Could not auto-find test image for class '{first_class_name}'. Please set 'test_image_path' manually below.")
              test_image_path = "/path/to/your/test/image.jpg" # Placeholder - MUST BE CHANGED MANUALLY
    else:
         print("Class names not available. Please set 'test_image_path' manually below.")
         test_image_path = "/path/to/your/test/image.jpg" # Placeholder - MUST BE CHANGED MANUALLY

    # <<< --- Make sure test_image_path is valid before proceeding --- >>>
    print(f"Attempting static inference on: {test_image_path}")

    if os.path.exists(test_image_path):
        print(f"\n--- Running Inference on single image: {os.path.basename(test_image_path)} ---")
        if not gallery:
             print("Skipping single image inference: Gallery is empty.")
        elif not haar_cascade_available:
             print("Skipping single image inference: Haar cascade XML not available.")
        else:
             _, recognition_results = recognize_faces_in_image(
                 inference_model,
                 gallery,
                 inference_transform,
                 CONFIG["device"],
                 test_image_path,
                 CONFIG["haar_cascade_path"],
                 CONFIG["recognition_threshold"],
                 CONFIG # Pass full config
             )
             if recognition_results:
                 print("\nRecognition Results (Single Image):")
                 for res in recognition_results:
                     print(f" - Box: {res['box']}, Identity: {res['identity']}, Score: {res['score']:.3f}")
             else:
                 print("\nNo recognition results returned (perhaps no faces detected or error occurred).")
    else:
        print(f"\nSkipping single image inference: Test image path not found or invalid: {test_image_path}")
        print("--> Please provide a valid path to an image file accessible by the notebook.")


    # --- Example: Inference on Validation Set Samples (Tensor-based) ---
    if val_loader and gallery:
        infer_on_validation(
            inference_model,
            gallery,
            val_loader,
            CONFIG["device"],
            CONFIG["recognition_threshold"],
            CONFIG, # Pass full config
            num_images=10 # Number of validation images to test
        )
    elif not gallery:
        print("\nSkipping validation set sample inference: Gallery is empty.")
    elif val_loader is None:
        print("\nSkipping validation set sample inference: Validation loader not available.")


print("\n--- Full Script Execution Finished ---")
