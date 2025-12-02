from collections import Counter
import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_ecg
from src.embedding import optimal_tau, embed
from src.multifractal import multifractal_spectrum, subsample
from src.features import extract_alpha0, extract_alpha_edges, compute_slopes
from src.classification import train_svc


# CONFIG
DATA_DIR = "data/ptbdb"
M = 4
TAU = 198
Q_VALS = np.linspace(-5, 5, 51)


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


# FEATURE EXTRACTION
X = []
y = []

for patient, record in patient_to_record.items():

    full_path = f"{DATA_DIR}/{record}"

    try:
        # no channel 6 -> skip automatically
        sig, meta = wfdb.rdsamp(full_path, channels=[6])
        sig = sig.flatten()
        fs = meta['fs']

        # preprocess
        sig_p = preprocess_ecg(sig, fs)

        # embed
        Y = embed(sig_p, M, TAU)
        Y_sub = subsample(Y)

        # multifractal spectrum
        alpha, f = multifractal_spectrum(Y_sub, Q_VALS)

        # extract features
        a0 = extract_alpha0(alpha, f)
        a1, a2 = extract_alpha_edges(alpha, f)
        width = a2 - a1
        g1, g2 = compute_slopes(alpha, f)

        X.append([a0, a1, a2, width, g1, g2])

        # label
        label = 0 if patient in healthy_list else 1
        y.append(label)

    except Exception as e:
        print("Error processing", patient, "->", e)


X = np.array(X)
y = np.array(y)

print("Extracted dataset shape:", X.shape)
label_counts = Counter(y.tolist())
print("Label distribution:", label_counts)


# CLASSIFICATION
model, acc, cm = train_svc(X, y)

if model is None:
    print("Training was skipped because the label set contained fewer than two classes or training split was degenerate.")
else:
    print("\nAccuracy:", acc)
    print("Confusion matrix:\n", cm)
