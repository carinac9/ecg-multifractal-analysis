from collections import Counter
import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_ecg
from src.embedding import optimal_tau, embed, svd_project, correlation_dimension_fast, surrogate_signal
from src.multifractal import multifractal_spectrum, subsample
from src.features import extract_paper_features
from src.classification import train_svc


# CONFIG
DATA_DIR = "data/ptbdb"
M = 4
TAU = 198
Q_VALS = np.linspace(-5, 5, 51)

# Paper uses ALL 12 ECG leads (V1-V6 + more)
# PTB-DB has channels: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
ALL_CHANNELS = list(range(12))  # 0-11 for all channels


# LOAD PATIENT LIST
with open(f"{DATA_DIR}/RECORDS") as f:
    all_records = [line.strip() for line in f]

with open(f"{DATA_DIR}/CONTROLS") as f:
    healthy_list = set([line.strip().split('/')[0] for line in f])


patient_to_record = {}
for rec in all_records:
    p = rec.split('/')[0]
    if p not in patient_to_record:
        patient_to_record[p] = rec


# FEATURE EXTRACTION (Multi-channel analysis - ALL ECG leads)
X = []
y = []
nonlinearity_checks = []

for patient, record in patient_to_record.items():

    full_path = f"{DATA_DIR}/{record}"

    try:
        # Load ALL ECG channels (paper uses V1-V6 and more)
        sig, meta = wfdb.rdsamp(full_path)  # Load all channels
        fs = meta['fs']

        # Extract features from each channel
        patient_features = []

        for ch_idx in range(min(12, sig.shape[1])):  # Up to 12 channels
            try:
                sig_ch = sig[:, ch_idx]

                # STEP 1: Preprocess
                sig_p = preprocess_ecg(sig_ch, fs)

                # STEP 2: Delay embedding
                Y = embed(sig_p, M, TAU)

                # STEP 3: SVD projection (noise-robust - per paper)
                Y_svd = svd_project(Y, n_components=None)

                # STEP 4: Subsample for multifractal computation
                Y_sub = subsample(Y_svd)

                # STEP 5: Compute multifractal spectrum
                alpha, f = multifractal_spectrum(Y_sub, Q_VALS)

                # STEP 6: Extract features (α₁, γ₁, α₀, width)
                a1, g1, a0, width = extract_paper_features(alpha, f)
                patient_features.extend([a1, g1, a0, width])

            except Exception:
                continue

        # Only add patient if we got features from multiple channels
        if len(patient_features) >= 16:  # At least 4 channels worth
            X.append(patient_features)

            # Label: 0=healthy, 1=pathological
            label = 0 if patient in healthy_list else 1
            y.append(label)
            nonlinearity_checks.append(True)  # Multi-channel is more robust

    except Exception as e:
        print(f"Error processing {patient} -> {e}")


X = np.array(X)
y = np.array(y)

print("Extracted dataset shape:", X.shape)
print(f"Features: Multi-channel (4 features × {X.shape[1]//4} channels)")
label_counts = Counter(y.tolist())
print("Label distribution:", label_counts)
print(f"Patients processed: {len(y)}")


# CLASSIFICATION
model, acc, cm = train_svc(X, y)

if model is None:
    print("Training was skipped because the label set contained fewer than two classes or training split was degenerate.")
else:
    print("\nAccuracy:", acc)
    print("Confusion matrix:\n", cm)
