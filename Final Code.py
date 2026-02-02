# ============================================================
# MM-CXR-CLIN-Attention-DenseNet121 + XAI (FIXED FINAL VERSION)
# ============================================================

import os, cv2, copy
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc, accuracy_score, recall_score
)

from scipy.stats import pearsonr, spearmanr

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"D:\project\data\labels.csv"
XRAY_DIR = r"D:\project\data\xrays"

ID_COL = "to_patient_id"
TARGET = "invasive_vent_days"

BATCH_SIZE = 8
EPOCHS = 120
LR = 1e-4
WEIGHT_DECAY = 1e-5
FOLDS = 5
PATIENCE = 20

VENT_THRESHOLD = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 Device:", DEVICE)

# ============================================================
# LOAD & FILTER DATA
# ============================================================

df = pd.read_csv(CSV_PATH)
df = df[df[TARGET].notna()].reset_index(drop=True)

def has_dicom(pid):
    pid = str(pid).strip()
    return os.path.exists(os.path.join(XRAY_DIR, f"{pid}.dcm"))

df["has_dicom"] = df[ID_COL].apply(has_dicom)
df = df[df["has_dicom"]].reset_index(drop=True)

print("✅ Valid samples with DICOM:", len(df))

# ============================================================
# TARGET + AUX FEATURES
# ============================================================

df["target_norm"] = np.log1p(df[TARGET].astype(float))
df["was_ventilated"] = (df[TARGET] > 0).astype(int)

# ============================================================
# CLINICAL FEATURES
# ============================================================

exclude = [ID_COL, TARGET, "target_norm", "has_dicom", "was_ventilated"]
clinical_cols = [c for c in df.columns if c not in exclude]

# force numeric
df[clinical_cols] = df[clinical_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# standardize
scaler = StandardScaler()
df[clinical_cols] = scaler.fit_transform(df[clinical_cols])

# ensure float32 columns
df[clinical_cols] = df[clinical_cols].astype(np.float32)

# ============================================================
# DATASET
# ============================================================

class MultimodalDataset(Dataset):
    def __init__(self, df_in, augment=False):
        self.df = df_in.reset_index(drop=True)

        tf = []
        if augment:
            tf += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10)
            ]
        tf += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ]
        self.transform = transforms.Compose(tf)

    def load_dicom(self, pid):
        pid = str(pid).strip()
        path = os.path.join(XRAY_DIR, f"{pid}.dcm")

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # normalize 0-1 safely
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # resize + make 3-channel
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        img = np.stack([img]*3, axis=-1)  # (H,W,3)

        return self.transform(Image.fromarray((img*255).astype(np.uint8)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # IMPORTANT: avoid using a mixed-type row Series for clinical values
        pid = self.df.at[idx, ID_COL]
        img = self.load_dicom(pid)

        clin_np = self.df.loc[idx, clinical_cols].to_numpy(dtype=np.float32)  # ✅ FIX
        clin = torch.from_numpy(clin_np)

        y = np.float32(self.df.at[idx, "target_norm"])
        y = torch.tensor(y, dtype=torch.float32)

        return img, clin, y

# ============================================================
# ATTENTION FUSION
# ============================================================

class AttentionFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d*2, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        w = self.fc(torch.cat([a, b], dim=1))
        return w*a + (1-w)*b

# ============================================================
# MODEL
# ============================================================

class MMNet(nn.Module):
    def __init__(self, clin_dim):
        super().__init__()

        base = models.densenet121(weights="DEFAULT")

        # freeze early parameters (your original approach)
        for p in list(base.parameters())[:200]:
            p.requires_grad = False

        self.cnn = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.img_fc = nn.Linear(1024, 512)
        self.clin_fc = nn.Sequential(
            nn.Linear(clin_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.att = AttentionFusion(512)

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img, clin):
        x = self.pool(self.cnn(img)).flatten(1)
        img_feat = self.img_fc(x)
        clin_feat = self.clin_fc(clin)
        fused = self.att(img_feat, clin_feat)
        return self.regressor(fused).squeeze(1)

# ============================================================
# TRAINING (K-FOLD)
# ============================================================

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

all_true, all_pred = [], []
fold_train_curves, fold_val_curves, lr_curve = [], [], []

for fold, (tr, va) in enumerate(kf.split(df)):

    print(f"\n========== FOLD {fold+1} ==========")

    tr_loader = DataLoader(
        MultimodalDataset(df.iloc[tr], augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    va_loader = DataLoader(
        MultimodalDataset(df.iloc[va], augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = MMNet(len(clinical_cols)).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

    loss_fn = nn.SmoothL1Loss()

    best_loss = 1e9
    wait = 0
    tr_curve, va_curve = [], []

    for epoch in range(EPOCHS):

        model.train()
        s = 0.0

        for img, clin, y in tr_loader:
            img, clin, y = img.to(DEVICE), clin.to(DEVICE), y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            pred = model(img, clin)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            s += loss.item()

        tr_loss = s / max(1, len(tr_loader))
        tr_curve.append(tr_loss)
        lr_curve.append(opt.param_groups[0]["lr"])

        model.eval()
        s = 0.0
        with torch.no_grad():
            for img, clin, y in va_loader:
                img, clin, y = img.to(DEVICE), clin.to(DEVICE), y.to(DEVICE)
                pred = model(img, clin)
                s += loss_fn(pred, y).item()

        va_loss = s / max(1, len(va_loader))
        va_curve.append(va_loss)

        sch.step(va_loss)

        print(f"Epoch {epoch+1:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss < best_loss:
            best_loss = va_loss
            best_model = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⏹ Early stopping")
                break

    model.load_state_dict(best_model)
    fold_train_curves.append(tr_curve)
    fold_val_curves.append(va_curve)

    model.eval()
    with torch.no_grad():
        for img, clin, y in va_loader:
            pred = model(img.to(DEVICE), clin.to(DEVICE))
            all_true.extend(np.expm1(y.numpy()))
            all_pred.extend(np.expm1(pred.cpu().numpy()))

# ============================================================
# FINAL METRICS
# ============================================================

all_true = np.array(all_true)
all_pred = np.array(all_pred)

print("\n========== FINAL RESULTS ==========")
print("MAE :", mean_absolute_error(all_true, all_pred))
print("RMSE:", np.sqrt(mean_squared_error(all_true, all_pred)))
print("R²  :", r2_score(all_true, all_pred))
print("PCC :", pearsonr(all_true, all_pred)[0])
print("SCC :", spearmanr(all_true, all_pred)[0])

# ============================================================
# BINARY CLASSIFICATION (>7 days)
# ============================================================

true_bin = (all_true > VENT_THRESHOLD).astype(int)
pred_bin = (all_pred > VENT_THRESHOLD).astype(int)

print("\nBinary metrics (>7 days)")
print("Accuracy   :", accuracy_score(true_bin, pred_bin))
print("Sensitivity:", recall_score(true_bin, pred_bin))
print("Specificity:", recall_score(true_bin, pred_bin, pos_label=0))

# ============================================================
# ROC
# ============================================================

fpr, tpr, _ = roc_curve(true_bin, all_pred)
plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(); plt.grid(); plt.show()

# ============================================================
# TRUE vs PREDICTED
# ============================================================

plt.scatter(all_true, all_pred, alpha=0.5)
m = max(all_true.max(), all_pred.max())
plt.plot([0, m], [0, m], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("True vs Predicted Ventilation Days")
plt.grid(); plt.show()

# ============================================================
# BLAND–ALTMAN
# ============================================================

mean_vals = (all_true + all_pred) / 2
diff_vals = (all_pred - all_true)
md, sd = diff_vals.mean(), diff_vals.std()

plt.scatter(mean_vals, diff_vals, alpha=0.5)
plt.axhline(md)
plt.axhline(md + 1.96*sd)
plt.axhline(md - 1.96*sd)
plt.xlabel("Mean"); plt.ylabel("Difference")
plt.title("Bland–Altman Plot")
plt.grid(); plt.show()

# ============================================================
# LEARNING & LR CURVES
# ============================================================

min_len = min(len(c) for c in fold_train_curves)
train_mean = np.mean([c[:min_len] for c in fold_train_curves], axis=0)
val_mean   = np.mean([c[:min_len] for c in fold_val_curves], axis=0)

plt.plot(train_mean, label="Train")
plt.plot(val_mean, label="Validation")
plt.legend(); plt.grid()
plt.title("Average Learning Curve")
plt.show()

plt.plot(lr_curve)
plt.title("Learning Rate Schedule")
plt.grid(); plt.show()
