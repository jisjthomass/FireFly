"""
train.py — Full-Brain Multi-Class Training Pipeline v3.1
==========================================================
Trains FlyBrainNet on the complete biological connectome using
the NSL-KDD dataset with 5-class attack categorization.

v3.1 Changes:
  • Focal Loss (γ=2) to address extreme class imbalance
  • SMOTE oversampling for R2L and U2R minority classes
  • Epoch-by-epoch training history saved for dashboard charts
  • Synaptic weight deltas saved for plasticity visualization

Classes:
  0 = Normal    (benign traffic)
  1 = DoS       (denial of service)
  2 = Probe     (port scanning / surveillance)
  3 = R2L       (remote-to-local unauthorized access)
  4 = U2R       (user-to-root privilege escalation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE

from model import FlyBrainNet


# ── Focal Loss for class imbalance ──
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — down-weights easy examples so the model
    focuses on hard, misclassified minority classes like R2L and U2R.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ── Attack category mapping (NSL-KDD) ──
ATTACK_CATEGORIES = {
    "normal": 0,
    # DoS attacks
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "mailbomb": 1, "apache2": 1, "processtable": 1,
    "udpstorm": 1,
    # Probe attacks
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2,
    "mscan": 2, "saint": 2,
    # R2L attacks
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3,
    "phf": 3, "spy": 3, "warezclient": 3, "warezmaster": 3,
    "xlock": 3, "xsnoop": 3, "snmpguess": 3, "snmpgetattack": 3,
    "httptunnel": 3, "sendmail": 3, "named": 3, "worm": 3,
    # U2R attacks
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "xterm": 4, "ps": 4, "sqlattack": 4,
}

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]


def load_and_preprocess_data():
    """Load NSL-KDD and encode features with 5-class labels."""
    print("[1/4] Loading NSL-KDD dataset...")
    df = pd.read_csv("KDDTrain.txt", names=COLUMNS)

    # Map attack names to 5 categories
    df["attack_class"] = df["label"].map(
        lambda x: ATTACK_CATEGORIES.get(x.strip().lower(), 0)
    )

    # Print class distribution
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = (df["attack_class"] == cls_id).sum()
        print(f"       {cls_name:>8}: {count:>6,} samples")

    y = df["attack_class"].values
    df_features = df.drop(["label", "difficulty_level", "attack_class"], axis=1)
    df_encoded = pd.get_dummies(df_features, columns=["protocol_type", "service", "flag"])

    print(f"       Total: {len(df):,} packets, {df_encoded.shape[1]} features")
    return df_encoded.values, y, df_encoded.columns, df


def train() -> None:
    """Full training pipeline with Focal Loss, SMOTE, and metrics."""
    print("═══════════════════════════════════════════════")
    print("  Biological Firewall — Training Pipeline v3.1")
    print("  (Focal Loss + SMOTE + Synaptic Delta Tracking)")
    print("═══════════════════════════════════════════════\n")

    X, y, feature_columns, df_raw = load_and_preprocess_data()

    # ── Split data ──
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ── Scale data (fit on training set ONLY) ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # ── SMOTE oversampling for minority classes ──
    print("\n[1.5/4] Applying SMOTE oversampling for R2L and U2R...")
    original_counts = np.bincount(y_train, minlength=5)
    print(f"       Before SMOTE: {dict(zip(CLASS_NAMES, original_counts))}")
    
    smote = SMOTE(
        sampling_strategy={
            3: max(original_counts[3] * 10, 2000),  # R2L: boost to ~2000
            4: max(original_counts[4] * 20, 1000),   # U2R: boost to ~1000
        },
        random_state=42,
        k_neighbors=min(5, original_counts[4] - 1),  # Ensure k < minority count
    )
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    new_counts = np.bincount(y_train, minlength=5)
    print(f"       After SMOTE:  {dict(zip(CLASS_NAMES, new_counts))}")
    print(f"       Total training samples: {len(y_train):,}")

    # ── Load the sparse connectome ──
    print("\n[2/4] Loading biological connectome...")
    connectome_path = "sparse_connectome.pt"
    if not os.path.exists(connectome_path):
        print("       ERROR: sparse_connectome.pt not found. Run fetch_connectome.py first.")
        return

    sparse_adj = torch.load(connectome_path, weights_only=False)
    num_neurons = sparse_adj.shape[0]
    nnz = sparse_adj._nnz()
    print(f"       ✓ {num_neurons:,} neurons, {nnz:,} synapses (sparse)")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # ── Build model ──
    print("\n[3/4] Building FlyBrainNet...")
    input_features = X.shape[1]
    num_classes = 5

    model = FlyBrainNet(input_features, num_neurons, num_classes, sparse_adj)
    
    if os.path.exists("fly_brain_weights.pt"):
        os.remove("fly_brain_weights.pt")
        print("       [!] Deleted old fly_brain_weights.pt for clean biologically-trained weights.")

    # ── Save initial synaptic weights for delta tracking ──
    initial_edge_weights = model.synaptic_layer.edge_weights.clone().detach()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total parameters:     {total_params:,}")
    print(f"       Trainable parameters: {trainable_params:,}")

    # ── Training config — Focal Loss ──
    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    alpha = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    print(f"       Loss: Focal Loss (γ=2.0)")
    print(f"       Class weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    epochs = 10
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batch_size = 512
    val_batch_size = 1024

    # ── Training loop ──
    print(f"\n[4/4] Starting training for {epochs} epochs...")
    print("─" * 70)

    best_val_acc = 0.0
    best_f1 = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        correct, total, epoch_loss = 0, 0, 0.0

        # Reset brain_state at the start of each epoch
        brain_state = None

        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            if brain_state is not None and brain_state.size(0) != bx.size(0):
                brain_state = None
            out, current_brain_state = model(bx, brain_state)
            
            brain_state = current_brain_state.detach()

            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bx.size(0)
            _, pred = torch.max(out, 1)
            total += by.size(0)
            correct += (pred == by).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = epoch_loss / total

        # ── Validation ──
        model.eval()
        val_preds = []
        with torch.no_grad():
            val_brain_state = None
            for i in range(0, X_val_t.size(0), val_batch_size):
                bx_val = X_val_t[i : i + val_batch_size]
                if val_brain_state is not None and val_brain_state.size(0) != bx_val.size(0):
                    val_brain_state = None
                val_out, current_val_brain_state = model(bx_val, val_brain_state)
                val_brain_state = current_val_brain_state.detach()
                
                _, v_pred = torch.max(val_out, 1)
                val_preds.append(v_pred)
                
            val_pred = torch.cat(val_preds)
            val_acc = 100.0 * (val_pred == y_val_t).sum().item() / len(y_val_t)
            epoch_f1 = f1_score(y_val_t.numpy(), val_pred.numpy(), average="macro")

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:>2}/{epochs} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Train: {train_acc:.2f}% │ "
            f"Val: {val_acc:.2f}% │ "
            f"F1: {epoch_f1:.4f} │ "
            f"LR: {lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "f1_macro": epoch_f1,
            "lr": lr
        })

        # Save best model by F1 (more important than raw accuracy)
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fly_brain_weights.pt")

    print("─" * 70)

    # ── Verify Biological Learning & Save Deltas ──
    final_edge_weights = model.synaptic_layer.edge_weights.clone().detach()
    weight_diff = torch.abs(final_edge_weights - initial_edge_weights).mean().item()
    if weight_diff > 1e-7:
        print(f"  ✓ Biological synapses successfully trained (mean weight change: {weight_diff:.6f})")
    else:
        print(f"  ✗ WARNING: Biological synapses did NOT learn (mean weight change: {weight_diff:.6f})")

    # Save synaptic weight deltas for visualization
    synaptic_deltas = {
        "initial_weights": initial_edge_weights,
        "final_weights": final_edge_weights,
        "weight_deltas": (final_edge_weights - initial_edge_weights),
    }
    torch.save(synaptic_deltas, "synaptic_deltas.pt")
    print("  ✓ Saved synaptic_deltas.pt (plasticity visualization data)")

    # ── Final evaluation ──
    model.load_state_dict(torch.load("fly_brain_weights.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        val_preds = []
        val_brain_state = None
        for i in range(0, X_val_t.size(0), val_batch_size):
            bx_val = X_val_t[i : i + val_batch_size]
            if val_brain_state is not None and val_brain_state.size(0) != bx_val.size(0):
                val_brain_state = None
            val_out, current_val_brain_state = model(bx_val, val_brain_state)
            val_brain_state = current_val_brain_state.detach()
            
            _, v_pred = torch.max(val_out, 1)
            val_preds.append(v_pred)
        val_pred = torch.cat(val_preds)

    y_true = y_val_t.numpy()
    y_pred = val_pred.numpy()

    print("\n═══════════════════════════════════════════════")
    print("  EVALUATION RESULTS")
    print("═══════════════════════════════════════════════\n")

    final_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Best F1 Macro:            {final_f1:.4f}\n")
    print(
        classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>8}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>8}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>8}", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()

    # ── Save training metadata ──
    metrics = {
        "best_val_accuracy": best_val_acc,
        "f1_macro": final_f1,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
        ),
        "num_neurons": num_neurons,
        "num_synapses": nnz,
        "epochs": epochs,
        "input_features": input_features,
        "num_classes": num_classes,
        "history": history,
        "smote_applied": True,
        "focal_loss_gamma": 2.0,
    }
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n  ✓ Saved training_metrics.json")

    # Save the scaler for consistent inference
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  ✓ Saved scaler.pkl")

    # Save feature columns
    torch.save(feature_columns, "feature_columns.pt")
    print("  ✓ Saved feature_columns.pt")

    print("\n═══════════════════════════════════════════════")
    print("  Training complete.")
    print("═══════════════════════════════════════════════")


if __name__ == "__main__":
    train()
