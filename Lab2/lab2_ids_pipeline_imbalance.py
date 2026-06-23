"""
LAB 2: Security Data Pipeline + IDS + Imbalance Handling
=========================================================
Dataset: CICIDS2017 (Monday — benign-only baseline + smaller labeled subset)
         We use the "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
         which contains Benign + PortScan + DDoS + Bot traffic.

Download the CICIDS2017 CSV files from:
  https://www.unb.ca/cic/datasets/ids-2017.html

  Direct mirror (Kaggle):
  https://www.kaggle.com/datasets/cicdataset/cicids2017

  Recommended file: "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
  Found in: Material/Lab2/Datasets/MachineLearningCVE/ folder

This lab builds a complete IDS pipeline:
  1. Preprocessing & feature engineering
  2. IDS model training (Logistic Regression + Random Forest)
  3. Multi-metric evaluation
  4. Imbalance handling (class_weight, undersampling) comparison
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Datasets", "MachineLearningCVE", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title, char="=", width=72):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def evaluate_model(name, y_true, y_pred, y_prob=None, class_names=None):
    """Compute and print a comprehensive set of evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  [{name}]")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Precision (wt):    {prec_w:.4f}")
    print(f"  Recall (wt):       {rec_w:.4f}")
    print(f"  F1 weighted:       {f1_w:.4f}")
    print(f"  F1 macro:          {f1_mac:.4f}")

    if y_prob is not None and len(np.unique(y_true)) == 2:
        auc_val = roc_auc_score(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        print(f"  ROC-AUC:           {auc_val:.4f}")
        print(f"  Avg Precision (PR):{ap_val:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    if class_names is not None:
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    return {
        "name": name, "accuracy": acc,
        "precision_w": prec_w, "recall_w": rec_w,
        "f1_weighted": f1_w, "f1_macro": f1_mac,
        "cm": cm, "y_pred": y_pred,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD RAW DATA
# ═════════════════════════════════════════════════════════════════════════════
print_header("LAB 2: SECURITY DATA PIPELINE + IDS + IMBALANCE")

if not os.path.exists(DATA_FILE):
    print(f"\n  ERROR: Dataset not found at:\n    {DATA_FILE}")
    print("""
  The dataset should be located in:
    Material/Lab2/Datasets/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  
  If the file is not available, download the CICIDS2017 dataset from:
    https://www.unb.ca/cic/datasets/ids-2017.html
  or Kaggle mirror:
    https://www.kaggle.com/datasets/cicdataset/cicids2017
""")
    sys.exit(1)

print("\n  Loading dataset...")
df_raw = pd.read_csv(DATA_FILE, low_memory=False)

# Strip whitespace from column names (CICIDS2017 has trailing spaces)
df_raw.columns = df_raw.columns.str.strip()
print(f"  Raw records: {len(df_raw):,}")
print(f"  Raw columns: {df_raw.shape[1]}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2: PREPROCESSING & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 1: PREPROCESSING & FEATURE ENGINEERING")

# --- 2a. Inspect the label column ---
label_col = "Label"
print(f"\n  --- Raw Label Distribution ---")
raw_counts = df_raw[label_col].value_counts()
for lbl, cnt in raw_counts.items():
    pct = cnt / len(df_raw) * 100
    print(f"    {lbl:<25} {cnt:>8,}  ({pct:5.1f}%)")

# --- 2b. Binary mapping: BENIGN=0, any attack=1 ---
df_raw["label_binary"] = (df_raw[label_col].str.strip() != "BENIGN").astype(int)

print(f"\n  Binary target: Normal={int((df_raw['label_binary']==0).sum()):,}, "
      f"Attack={int((df_raw['label_binary']==1).sum()):,}")

imbalance_ratio = (df_raw["label_binary"] == 0).sum() / max((df_raw["label_binary"] == 1).sum(), 1)
print(f"  Imbalance ratio (Normal:Attack) ≈ {imbalance_ratio:.1f}:1")

# --- 2c. Clean numeric columns ---
print("\n  Cleaning data...")

# Drop non-numeric / ID columns that leak info
drop_cols_if_exist = ["Flow ID", "Source IP", "Source Port",
                      "Destination IP", "Destination Port", "Timestamp",
                      label_col]
for col in drop_cols_if_exist:
    if col in df_raw.columns:
        df_raw.drop(columns=[col], inplace=True)

# Convert all feature columns to numeric
feature_cols = [c for c in df_raw.columns if c not in ["label_binary"]]
for col in feature_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Replace inf and -inf with NaN, then fill with 0
df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
nan_before = df_raw[feature_cols].isna().sum().sum()
df_raw[feature_cols] = df_raw[feature_cols].fillna(0)
print(f"  Inf/NaN values replaced: {nan_before:,}")

# Remove constant features (zero variance)
variances = df_raw[feature_cols].var()
constant_cols = variances[variances == 0].index.tolist()
if constant_cols:
    df_raw.drop(columns=constant_cols, inplace=True)
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    print(f"  Removed {len(constant_cols)} constant features: {constant_cols[:5]}...")

# --- 2d. Feature engineering ---
print("\n  --- Feature Engineering ---")
original_count = len(feature_cols)

# Ratio features (safe division)
if "Flow Duration" in df_raw.columns and "Total Fwd Packets" in df_raw.columns:
    df_raw["packets_per_duration"] = df_raw["Total Fwd Packets"] / (df_raw["Flow Duration"] + 1)
    feature_cols.append("packets_per_duration")

if "Total Length of Fwd Packets" in df_raw.columns and "Total Fwd Packets" in df_raw.columns:
    df_raw["avg_fwd_packet_size"] = (df_raw["Total Length of Fwd Packets"] /
                                      (df_raw["Total Fwd Packets"] + 1))
    feature_cols.append("avg_fwd_packet_size")

if "Flow Bytes/s" in df_raw.columns and "Flow Packets/s" in df_raw.columns:
    df_raw["bytes_per_packet"] = df_raw["Flow Bytes/s"] / (df_raw["Flow Packets/s"] + 1)
    feature_cols.append("bytes_per_packet")

new_features = len(feature_cols) - original_count
print(f"  Original features: {original_count}")
print(f"  Engineered features: {new_features}")
print(f"  Total features: {len(feature_cols)}")

# Remove duplicates
df_raw = df_raw.drop_duplicates()
print(f"\n  Records after dedup: {len(df_raw):,}")

# --- 2e. Final feature summary ---
print(f"\n  --- Feature Summary (top 10 by variance) ---")
feat_var = df_raw[feature_cols].var().sort_values(ascending=False).head(10)
for feat, var in feat_var.items():
    print(f"    {feat:<40} var={var:>12.2f}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3: TRAIN / TEST SPLIT + SCALING
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 2: TRAIN / TEST SPLIT + SCALING")

X = df_raw[feature_cols].values
y = df_raw["label_binary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\n  Train: {len(X_train):,} (Attack: {y_train.sum():,} = {y_train.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test):,}  (Attack: {y_test.sum():,} = {y_test.mean()*100:.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4: BASELINE IDS MODELS (NO IMBALANCE HANDLING)
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 3: BASELINE IDS MODELS (no imbalance handling)")

baseline_results = {}

# --- Logistic Regression (baseline) ---
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train_s, y_train)
y_pred_lr = lr_base.predict(X_test_s)
y_prob_lr = lr_base.predict_proba(X_test_s)[:, 1]
baseline_results["LR_baseline"] = evaluate_model(
    "Logistic Regression (baseline)", y_test, y_pred_lr, y_prob_lr,
    class_names=["Normal", "Attack"])

# --- Random Forest (baseline) ---
rf_base = RandomForestClassifier(n_estimators=100, max_depth=15,
                                  random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_rf = rf_base.predict(X_test)
y_prob_rf = rf_base.predict_proba(X_test)[:, 1]
baseline_results["RF_baseline"] = evaluate_model(
    "Random Forest (baseline)", y_test, y_pred_rf, y_prob_rf,
    class_names=["Normal", "Attack"])

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5: IMBALANCE HANDLING — class_weight="balanced"
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 4: IMBALANCE HANDLING — class_weight='balanced'")

print("\n  class_weight='balanced' adjusts weights inversely proportional")
print("  to class frequencies, giving minority (attack) class more weight.")

cw_results = {}

# --- LR with class_weight ---
lr_cw = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_cw.fit(X_train_s, y_train)
y_pred_lr_cw = lr_cw.predict(X_test_s)
y_prob_lr_cw = lr_cw.predict_proba(X_test_s)[:, 1]
cw_results["LR_balanced"] = evaluate_model(
    "Logistic Regression (class_weight=balanced)", y_test, y_pred_lr_cw, y_prob_lr_cw,
    class_names=["Normal", "Attack"])

# --- RF with class_weight ---
rf_cw = RandomForestClassifier(n_estimators=100, max_depth=15,
                                class_weight="balanced", random_state=42, n_jobs=-1)
rf_cw.fit(X_train, y_train)
y_pred_rf_cw = rf_cw.predict(X_test)
y_prob_rf_cw = rf_cw.predict_proba(X_test)[:, 1]
cw_results["RF_balanced"] = evaluate_model(
    "Random Forest (class_weight=balanced)", y_test, y_pred_rf_cw, y_prob_rf_cw,
    class_names=["Normal", "Attack"])

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6: IMBALANCE HANDLING — RANDOM UNDERSAMPLING
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 5: IMBALANCE HANDLING — RANDOM UNDERSAMPLING")

print("\n  Undersampling reduces the majority class to match the minority class size.")

# Undersample majority class
np.random.seed(42)
idx_attack = np.where(y_train == 1)[0]
idx_normal = np.where(y_train == 0)[0]

n_attack = len(idx_attack)
n_normal = len(idx_normal)
print(f"  Before: Normal={n_normal:,}, Attack={n_attack:,}")

# Undersample normal to match attack count
idx_normal_under = np.random.choice(idx_normal, size=n_attack, replace=False)
idx_under = np.concatenate([idx_normal_under, idx_attack])
np.random.shuffle(idx_under)

X_train_under = X_train[idx_under]
X_train_under_s = X_train_s[idx_under]
y_train_under = y_train[idx_under]

print(f"  After:  Normal={int((y_train_under==0).sum()):,}, Attack={int((y_train_under==1).sum()):,}")
print(f"  Undersampled training size: {len(y_train_under):,}")

us_results = {}

# --- LR with undersampling ---
lr_us = LogisticRegression(max_iter=1000, random_state=42)
lr_us.fit(X_train_under_s, y_train_under)
y_pred_lr_us = lr_us.predict(X_test_s)
y_prob_lr_us = lr_us.predict_proba(X_test_s)[:, 1]
us_results["LR_undersample"] = evaluate_model(
    "Logistic Regression (undersampled)", y_test, y_pred_lr_us, y_prob_lr_us,
    class_names=["Normal", "Attack"])

# --- RF with undersampling ---
rf_us = RandomForestClassifier(n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1)
rf_us.fit(X_train_under, y_train_under)
y_pred_rf_us = rf_us.predict(X_test)
y_prob_rf_us = rf_us.predict_proba(X_test)[:, 1]
us_results["RF_undersample"] = evaluate_model(
    "Random Forest (undersampled)", y_test, y_pred_rf_us, y_prob_rf_us,
    class_names=["Normal", "Attack"])

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 7: COMPREHENSIVE COMPARISON — FINDING THE BEST
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 6: COMPREHENSIVE COMPARISON — FINDING THE BEST")

all_results = {}
all_results.update(baseline_results)
all_results.update(cw_results)
all_results.update(us_results)

# Summary table
metrics_list = ["accuracy", "precision_w", "recall_w", "f1_weighted", "f1_macro"]
header = f"  {'Model':<35}"
for m in ["Acc", "Prec(w)", "Rec(w)", "F1(w)", "F1(mac)"]:
    header += f" {m:>8}"
print(f"\n{header}")
print("  " + "-" * 80)

for key, res in all_results.items():
    row = f"  {key:<35}"
    for m in metrics_list:
        row += f" {res[m]:>8.4f}"
    print(row)

# Best by F1-macro (best metric for imbalanced data)
best_key = max(all_results, key=lambda k: all_results[k]["f1_macro"])
best_res = all_results[best_key]
print(f"\n  >>> BEST MODEL (by Macro F1): {best_key}")
print(f"      F1 Macro = {best_res['f1_macro']:.4f}, Accuracy = {best_res['accuracy']:.4f}")

# ─── Per-model confusion matrix analysis ───
print(f"\n  --- Confusion Matrix Analysis ---")
print(f"  {'Model':<35} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6}  {'FN%':>6} {'FP%':>6}")
print("  " + "-" * 78)

for key, res in all_results.items():
    cm = res["cm"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fn_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    print(f"  {key:<35} {tn:>6} {fp:>6} {fn:>6} {tp:>6}  {fn_rate:>5.1f}% {fp_rate:>5.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 8: SECURITY IMPACT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 7: SECURITY IMPACT ANALYSIS")

print("""
  In IDS, the cost of errors is asymmetric:
  - False Negative (FN): Missed attack → potential breach (HIGH COST)
  - False Positive (FP): False alarm → wasted analyst time (MODERATE COST)

  Therefore: Recall on the Attack class is the most critical metric.
""")

print(f"  {'Model':<35} {'Attack Recall':>14} {'Attack Prec':>12} {'Verdict':>12}")
print("  " + "-" * 76)

for key, res in all_results.items():
    cm = res["cm"]
    fn, tp = cm[1][0], cm[1][1]
    fp = cm[0][1]
    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    attack_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    if attack_recall >= 0.95:
        verdict = "✓ STRONG"
    elif attack_recall >= 0.85:
        verdict = "~ MODERATE"
    else:
        verdict = "✗ WEAK"
    print(f"  {key:<35} {attack_recall:>12.4f} {attack_prec:>12.4f} {verdict:>12}")

# Recommendation
print(f"\n  --- Recommendation ---")
# Find model with best attack recall among those with >85% accuracy
viable = {k: v for k, v in all_results.items() if v["accuracy"] > 0.85}
if viable:
    best_recall_key = max(viable, key=lambda k: (
        viable[k]["cm"][1][1] / max(viable[k]["cm"][1][0] + viable[k]["cm"][1][1], 1)))
    br = all_results[best_recall_key]
    cm_br = br["cm"]
    rec_val = cm_br[1][1] / max(cm_br[1][0] + cm_br[1][1], 1)
    print(f"  For deployment: {best_recall_key}")
    print(f"  Reason: Best attack recall ({rec_val:.4f}) while maintaining >85% accuracy")
else:
    print(f"  For deployment: {best_key} (best overall F1)")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 9: FEATURE IMPORTANCE (Random Forest)
# ═════════════════════════════════════════════════════════════════════════════
print_header("STEP 8: FEATURE IMPORTANCE (Random Forest)")

rf_for_imp = rf_base  # use baseline RF (trained on full data)
if hasattr(rf_for_imp, "feature_importances_"):
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_for_imp.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(f"\n  Top 20 Most Important Features for IDS:")
    for rank, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        bar = "█" * int(row["Importance"] * 80)
        print(f"  {rank:>3}. {row['Feature']:<40} {row['Importance']:.4f}  {bar}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 10: VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
print_header("GENERATING VISUALIZATIONS...")

# --- Figure 1: Confusion Matrices (all 6 models) ---
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

model_order = ["LR_baseline", "RF_baseline",
               "LR_balanced", "RF_balanced",
               "LR_undersample", "RF_undersample"]
titles = ["LR (baseline)", "RF (baseline)",
          "LR (class_weight=balanced)", "RF (class_weight=balanced)",
          "LR (undersampled)", "RF (undersampled)"]
cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlOrBr"]

for idx, (key, title, cmap) in enumerate(zip(model_order, titles, cmaps)):
    ax = axes1[idx // 3][idx % 3]
    res = all_results[key]
    ConfusionMatrixDisplay(res["cm"], display_labels=["Normal", "Attack"]).plot(
        ax=ax, cmap=cmap)
    acc_val = res["accuracy"]
    f1_val = res["f1_macro"]
    ax.set_title(f"{title}\nAcc={acc_val:.4f} | F1m={f1_val:.4f}", fontsize=9)

fig1.tight_layout()
fig1.savefig(os.path.join(SCRIPT_DIR, "fig1_confusion_matrices.png"), dpi=150)
plt.close(fig1)
print("  Saved: fig1_confusion_matrices.png")

# --- Figure 2: Metric Comparison Bar Chart ---
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Model Comparison — Metrics", fontsize=14, fontweight="bold")

# F1 macro comparison
names_short = ["LR\nbase", "RF\nbase", "LR\nbalanced", "RF\nbalanced", "LR\nunder", "RF\nunder"]
f1_vals = [all_results[k]["f1_macro"] for k in model_order]
acc_vals = [all_results[k]["accuracy"] for k in model_order]
colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#f39c12"]

x_pos = np.arange(len(model_order))
bars = axes2[0].bar(x_pos, f1_vals, color=colors, edgecolor="black", linewidth=0.5)
axes2[0].set_xticks(x_pos)
axes2[0].set_xticklabels(names_short, fontsize=8)
axes2[0].set_ylabel("F1 Macro")
axes2[0].set_title("F1 Macro Score (higher = better for imbalanced data)")
axes2[0].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, f1_vals):
    axes2[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8)

# FN (missed attacks) comparison
fn_vals = [all_results[k]["cm"][1][0] for k in model_order]
fp_vals = [all_results[k]["cm"][0][1] for k in model_order]

width = 0.35
axes2[1].bar(x_pos - width/2, fn_vals, width, label="FN (missed attacks)", color="#e74c3c")
axes2[1].bar(x_pos + width/2, fp_vals, width, label="FP (false alarms)", color="#f39c12")
axes2[1].set_xticks(x_pos)
axes2[1].set_xticklabels(names_short, fontsize=8)
axes2[1].set_ylabel("Count")
axes2[1].set_title("Error Analysis: Missed Attacks vs False Alarms")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3, axis="y")

fig2.tight_layout()
fig2.savefig(os.path.join(SCRIPT_DIR, "fig2_metric_comparison.png"), dpi=150)
plt.close(fig2)
print("  Saved: fig2_metric_comparison.png")

# --- Figure 3: Feature Importance + Imbalance Effect ---
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle("Feature Importance & Imbalance Handling Effect", fontsize=14, fontweight="bold")

# Feature importance (top 15)
if hasattr(rf_for_imp, "feature_importances_"):
    imp_top15 = imp_df.head(15).iloc[::-1]
    clrs = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_top15)))
    axes3[0].barh(imp_top15["Feature"], imp_top15["Importance"], color=clrs)
    axes3[0].set_xlabel("Importance")
    axes3[0].set_title("Top 15 Features (RF baseline)")
    axes3[0].tick_params(axis="y", labelsize=8)

# Recall vs Precision tradeoff across strategies
strategies = ["Baseline", "class_weight", "Undersample"]
lr_recalls = []
lr_precs = []
rf_recalls = []
rf_precs = []

for key_lr, key_rf in [("LR_baseline", "RF_baseline"),
                        ("LR_balanced", "RF_balanced"),
                        ("LR_undersample", "RF_undersample")]:
    cm_lr = all_results[key_lr]["cm"]
    cm_rf = all_results[key_rf]["cm"]
    lr_recalls.append(cm_lr[1][1] / max(cm_lr[1].sum(), 1))
    lr_precs.append(cm_lr[1][1] / max(cm_lr[1][1] + cm_lr[0][1], 1))
    rf_recalls.append(cm_rf[1][1] / max(cm_rf[1].sum(), 1))
    rf_precs.append(cm_rf[1][1] / max(cm_rf[1][1] + cm_rf[0][1], 1))

x_s = np.arange(len(strategies))
width = 0.2
axes3[1].bar(x_s - 1.5*width, lr_recalls, width, label="LR Recall", color="#3498db")
axes3[1].bar(x_s - 0.5*width, lr_precs, width, label="LR Precision", color="#85c1e9")
axes3[1].bar(x_s + 0.5*width, rf_recalls, width, label="RF Recall", color="#2ecc71")
axes3[1].bar(x_s + 1.5*width, rf_precs, width, label="RF Precision", color="#82e0aa")
axes3[1].set_xticks(x_s)
axes3[1].set_xticklabels(strategies)
axes3[1].set_ylabel("Score (Attack class)")
axes3[1].set_title("Attack Recall vs Precision by Strategy")
axes3[1].legend(fontsize=8)
axes3[1].grid(True, alpha=0.3, axis="y")

fig3.tight_layout()
fig3.savefig(os.path.join(SCRIPT_DIR, "fig3_features_and_imbalance.png"), dpi=150)
plt.close(fig3)
print("  Saved: fig3_features_and_imbalance.png")

# ═════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print_header("FINAL SUMMARY")

print(f"""
  Dataset: CICIDS2017 ({len(df_raw):,} records, {len(feature_cols)} features)
  Task:    Binary IDS — Normal vs Attack

  Models tested (6 configurations):
    Baseline:         LR, RF (no imbalance handling)
    class_weight:     LR, RF (balanced class weights)
    Undersampling:    LR, RF (majority class reduced)

  Best Model: {best_key}
    Accuracy:   {best_res['accuracy']:.4f}
    F1 Macro:   {best_res['f1_macro']:.4f}

  Key Findings:
  - Imbalance handling significantly affects attack detection recall
  - class_weight='balanced' is the easiest approach (no data modification)
  - Undersampling can improve recall but may reduce precision
  - Random Forest generally outperforms Logistic Regression on this data
  - Feature importance reveals which network features are most predictive

  Generated Figures:
  - fig1_confusion_matrices.png    (all 6 model confusion matrices)
  - fig2_metric_comparison.png     (F1 and error analysis bar charts)
  - fig3_features_and_imbalance.png (feature importance + recall/precision)
""")
