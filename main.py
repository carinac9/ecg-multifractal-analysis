from collections import Counter
import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_ecg
from src.embedding import optimal_tau, embed, svd_project
from src.multifractal import multifractal_spectrum, subsample
from src.features import extract_paper_features, beat_replication_variability
from src.classification import train_svc


# ============================================================================
# CONFIGURATION - PAPER PARAMETERS
# ============================================================================
DATA_DIR = "data/ptbdb"
M = 4                             # Embedding dimension (Shekatkar et al.)
TAU = 198                         # Time delay (hardcoded from paper)
Q_VALS = np.linspace(-5, 5, 51)  # Generalized dimensions

# Paper uses V1-V6 chest electrodes (6 channels)
# V1-V6 are typically indices 6-11 in PTB-DB standard format
V1_TO_V6_INDICES = [6, 7, 8, 9, 10, 11]  # Chest electrodes only (per paper)


# ============================================================================
# LOAD PATIENT DATA
# ============================================================================
with open(f"{DATA_DIR}/RECORDS") as f:
    all_records = [line.strip() for line in f]

with open(f"{DATA_DIR}/CONTROLS") as f:
    healthy_list = set([line.strip().split('/')[0] for line in f])


patient_to_record = {}
for rec in all_records:
    p = rec.split('/')[0]
    if p not in patient_to_record:
        patient_to_record[p] = rec


# ============================================================================
# FEATURE EXTRACTION - V1-V6 CHANNELS ONLY (Per Shekatkar et al.)
# ============================================================================
X = []
y = []

for patient, record in patient_to_record.items():
    full_path = f"{DATA_DIR}/{record}"

    try:
        sig, meta = wfdb.rdsamp(full_path)
        fs = meta['fs']
        sig_names = meta.get('sig_name', [])

        # Extract V1-V6 channel indices (verify by name if possible)
        channel_indices = V1_TO_V6_INDICES
        if len(sig_names) >= 12:
            # Try to find V1-V6 by name
            try:
                channel_indices = [i for i, name in enumerate(sig_names) if name.strip() in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
                if not channel_indices:
                    channel_indices = V1_TO_V6_INDICES
            except:
                channel_indices = V1_TO_V6_INDICES

        patient_features = []

        # Process only V1-V6 channels (per paper)
        for ch_idx in channel_indices:
            if ch_idx >= sig.shape[1]:
                continue

            try:
                sig_ch = sig[:, ch_idx]

                # Preprocess: detrend + low-pass filter + normalize
                sig_p = preprocess_ecg(sig_ch, fs)

                # Delay embedding (M=4, τ=198)
                Y = embed(sig_p, M, TAU)

                # SVD projection for noise reduction (per paper)
                Y_svd = svd_project(Y, n_components=None)

                # Subsample attractor
                Y_sub = subsample(Y_svd)

                # Multifractal spectrum computation
                alpha, f = multifractal_spectrum(Y_sub, Q_VALS)

                # Extract multifractal parameters: α₁, γ₁, α₀, width
                a1, g1, a0, width = extract_paper_features(alpha, f)
                patient_features.extend([a1, g1, a0, width])

                # Beat-replication variability (δα₁)
                delta_a1 = beat_replication_variability(sig_p, fs, alpha, f)
                patient_features.append(delta_a1)

            except Exception:
                continue

        # Only add if extracted from sufficient channels
        if len(patient_features) >= 20:  # At least 4 V1-V6 channels
            X.append(patient_features)

            # Label: 0=healthy, 1=pathological
            label = 0 if patient in healthy_list else 1
            y.append(label)

    except Exception as e:
        # Silently skip problematic patients
        pass


X = np.array(X)
y = np.array(y)

print("\n" + "="*70)
print("ECG MULTIFRACTAL CLASSIFICATION - SHEKATKAR ET AL. REPLICATION")
print("="*70)
print(f"Dataset shape: {X.shape} samples")
print(f"Channels used: V1-V6 (6 chest electrodes per paper)")
print(f"Features: {X.shape[1]} = 5 features × 6 channels + δα₁")
label_counts = Counter(y.tolist())
print(f"Label distribution: {dict(label_counts)}")
print(f"  - Healthy (0): {label_counts[0]} patients")
print(f"  - Pathological (1): {label_counts[1]} patients")
print("="*70 + "\n")

# ============================================================================
# CLASSIFICATION (Ensemble: RF + SVM)
# ============================================================================
model, acc, cm = train_svc(X, y)

if model is None:
    print("Training was skipped because the label set contained fewer than two classes or training split was degenerate.")
else:
    print("\nAccuracy:", acc)
    print("Confusion matrix:\n", cm)
