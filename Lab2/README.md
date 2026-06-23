# LAB 2: Security Data Pipeline + IDS + Imbalance Handling

## Overview

This lab builds a **complete Intrusion Detection System (IDS) pipeline** using a real-world network traffic dataset. It covers data preprocessing, feature engineering, model training, multi-metric evaluation, and compares different strategies for handling class imbalance — a critical challenge in security data where attacks are rare compared to normal traffic.

## Topics Covered

| Topic | Description |
|-------|-------------|
| **Preprocessing & feature engineering** | Cleaning CICIDS2017 data, handling inf/NaN, encoding, scaling, creating ratio features |
| **IDS model training (LR / RF)** | Logistic Regression and Random Forest for binary intrusion detection |
| **Multi-metric evaluation** | Accuracy, precision, recall, F1 (weighted + macro), confusion matrix, ROC-AUC |
| **Imbalance handling — class_weight** | Using `class_weight='balanced'` to re-weight the loss function |
| **Imbalance handling — undersampling** | Random undersampling of majority class to create balanced training set |
| **Model comparison** | Side-by-side comparison of 6 model configurations to find the best |

## Dataset: CICIDS2017

The **CICIDS2017** dataset was created by the Canadian Institute for Cybersecurity and contains realistic network traffic with labeled attacks captured over 5 days.

- **Source**: Canadian Institute for Cybersecurity (UNB)
- **Official page**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Kaggle mirror**: https://www.kaggle.com/datasets/cicdataset/cicids2017
- **File used**: `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- **Traffic types**: BENIGN + PortScan + DDoS + Bot
- **Features**: 78+ network flow features (packet counts, byte rates, flags, flow duration, etc.)

### Why CICIDS2017?

- Real captured network traffic (not simulated)
- Contains modern attack types relevant to current threats
- Well-documented and widely cited in IDS research
- Exhibits natural class imbalance (most traffic is benign)

## Lab Structure

The lab performs **8 steps** in a single script:

```
Step 1: Preprocessing & Feature Engineering
         → Clean inf/NaN, remove constants, encode, create ratio features

Step 2: Train/Test Split + Scaling
         → Stratified 80/20 split, StandardScaler for LR

Step 3: Baseline IDS Models
         → LR and RF with no imbalance handling

Step 4: Imbalance Handling — class_weight='balanced'
         → LR and RF with automatic class re-weighting

Step 5: Imbalance Handling — Random Undersampling
         → Reduce majority class, retrain LR and RF

Step 6: Comprehensive Comparison
         → 6-model comparison table (accuracy, precision, recall, F1)

Step 7: Security Impact Analysis
         → Missed attacks vs false alarms, deployment recommendation

Step 8: Feature Importance
         → Top features identified by Random Forest
```

## How to Run

### Step 1: Download the Dataset

Download the CICIDS2017 CSV files from one of these sources:

- **Official UNB page**:
  ```
  https://www.unb.ca/cic/datasets/ids-2017.html
  ```

- **Kaggle mirror** (requires Kaggle account):
  ```
  https://www.kaggle.com/datasets/cicdataset/cicids2017
  ```

- **Use provided lab dataset**: use the CSV file already provided in the `Lab2/Datasets/` folder.

Download the file **`Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`**, rename it to **`friday_traffic.csv`**, and place it in the `Lab2/` folder:

```
Lab2/
├── friday_traffic.csv              ← download and place here
├── lab2_ids_pipeline_imbalance.py
└── README.md
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 3: Run the Lab

```bash
cd Lab2
python lab2_ids_pipeline_imbalance.py
```

### Output

The script will:
1. Print detailed analysis to the console (8 steps of the pipeline)
2. Generate 3 visualization files:
   - `fig1_confusion_matrices.png` — All 6 model confusion matrices (2×3 grid)
   - `fig2_metric_comparison.png` — F1 scores and error counts (FN vs FP)
   - `fig3_features_and_imbalance.png` — Feature importance + recall/precision by strategy

## What You Will Learn

1. **Preprocessing real security data**: Handling dirty CSV data with inf values, NaN, duplicate records, and mixed types
2. **Feature engineering**: Creating meaningful ratio features from raw network statistics
3. **Baseline models**: How LR and RF perform out-of-the-box on imbalanced security data
4. **class_weight parameter**: A simple one-parameter fix for class imbalance, built into sklearn
5. **Random undersampling**: Manually balancing training data by reducing the majority class
6. **Multi-metric evaluation**: Why accuracy alone is misleading for imbalanced data — F1 macro, recall, and confusion matrix analysis
7. **Security-focused thinking**: Why false negatives (missed attacks) are costlier than false positives (false alarms) in IDS

## Key Concepts

### Why Class Imbalance Matters in IDS

In real network traffic, >95% of connections are typically benign. A naive model can achieve 95%+ accuracy by simply predicting everything as "Normal" — while missing every single attack.

### Imbalance Handling Strategies Compared

| Strategy | How it Works | Pros | Cons |
|----------|-------------|------|------|
| **Baseline** | No handling | Simple, fast | Biased toward majority class |
| **class_weight='balanced'** | Adjusts loss function weights | No data modification needed | May increase false positives |
| **Random undersampling** | Removes majority samples | Forces balanced learning | Loses potentially useful data |

### Evaluation Metrics for IDS

| Metric | What it Measures | Why it Matters |
|--------|-----------------|----------------|
| **Accuracy** | Overall correctness | Misleading when imbalanced |
| **Precision** | Of predicted attacks, how many are real | False alarm rate |
| **Recall** | Of real attacks, how many were caught | Missed attack rate |
| **F1 Macro** | Harmonic mean across classes | Best single metric for imbalanced data |
| **Confusion Matrix** | Full error breakdown | Shows exact FP/FN counts |

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Appendix: Alternative Datasets

The following real-world datasets can be used as drop-in replacements or for further experimentation. All contain labeled network traffic with natural class imbalance.

| Dataset | Source | Attack Types | Download |
|---------|--------|-------------|----------|
| **CSE-CIC-IDS2018** | CIC/UNB | Brute Force, Botnet, DDoS, Web attacks, Infiltration | https://www.unb.ca/cic/datasets/ids-2018.html |
| **UNSW-NB15** | UNSW Sydney | 9 types: Fuzzers, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms, Analysis | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| **NSL-KDD** | UNB | 4 categories: DoS, Probe, R2L, U2R (41 features, classic benchmark) | https://www.unb.ca/cic/datasets/nsl.html |
| **CIC-DDoS2019** | CIC/UNB | DDoS-focused: NTP, DNS, SNMP, LDAP reflection + exploitation-based | https://www.unb.ca/cic/datasets/ddos-2019.html |
| **CTU-13** | CTU Prague | 13 botnet captures — real C&C traffic mixed with normal | https://www.stratosphereips.org/datasets-ctu13 |
| **LITNET-2020** | Kaunas Univ. | 12 attack classes, 85 features, real academic network traffic | https://dataset.litnet.lt/index.php |
| **Bot-IoT** | UNSW Sydney | IoT-specific botnet and attack traffic | https://research.unsw.edu.au/projects/bot-iot-dataset |
| **TON_IoT** | UNSW Sydney | IoT/IIoT telemetry: scanning, DoS, ransomware, backdoor | https://research.unsw.edu.au/projects/toniot-datasets |

### Recommended Alternatives

- **Drop-in replacement**: **UNSW-NB15** or **CSE-CIC-IDS2018** — similar CSV format, comparable features, natural imbalance
- **Classic benchmark**: **NSL-KDD** — smaller, well-studied, easy to compare with published literature
- **Specialized focus**: **CIC-DDoS2019** (DDoS only) or **Bot-IoT** (IoT attacks)
