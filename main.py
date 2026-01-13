from collections import Counter
import numpy as np
import wfdb

from src.preprocessing import preprocess_ecg
from src.embedding import embed, svd_project
from src.multifractal import multifractal_spectrum, subsample
from src.features import extract_paper_features, beat_replication_variability
from src.classification import train_svc_cv

DATA_DIR = "data/ptbdb"
M = 4
TAU = 198
Q_VALS = np.linspace(-5, 5, 51)

V1_TO_V6_INDICES = [6, 7, 8, 9, 10, 11]


with open(f"{DATA_DIR}/RECORDS") as f:
    all_records = [line.strip() for line in f]

with open(f"{DATA_DIR}/CONTROLS") as f:
    healthy_list = set([line.strip().split('/')[0] for line in f])


patient_to_record = {}
for rec in all_records:
    p = rec.split('/')[0]
    if p not in patient_to_record:
        patient_to_record[p] = rec

X = []
y = []
skipped_patients = []
partial_channels = []

for patient, record in patient_to_record.items():
    full_path = f"{DATA_DIR}/{record}"

    try:
        sig, meta = wfdb.rdsamp(full_path)
        fs = meta['fs']
        sig_names = meta.get('sig_name', [])

        channel_indices = V1_TO_V6_INDICES
        if len(sig_names) >= 12:
            try:
                channel_indices = [i for i, name in enumerate(sig_names) if name.strip() in [
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
                if not channel_indices:
                    channel_indices = V1_TO_V6_INDICES
            except Exception:
                channel_indices = V1_TO_V6_INDICES

        patient_features = []
        successful_channels = 0

        for ch_idx in channel_indices:
            if ch_idx >= sig.shape[1]:
                continue

            try:
                sig_ch = sig[:, ch_idx]
                sig_p = preprocess_ecg(sig_ch, fs)
                Y = embed(sig_p, M, TAU)
                Y_svd = svd_project(Y, n_components=None)
                Y_sub = subsample(Y_svd)
                alpha, f = multifractal_spectrum(Y_sub, Q_VALS)
                a1, g1, a0, width = extract_paper_features(alpha, f)
                patient_features.extend([a1, g1, a0, width])
                delta_a1 = beat_replication_variability(sig_p, fs, alpha, f)
                patient_features.append(delta_a1)
                successful_channels += 1

            except Exception as e:
                pass

        if len(patient_features) >= 20:
            X.append(patient_features)
            label = 0 if patient in healthy_list else 1
            y.append(label)

            if successful_channels < 6:
                partial_channels.append((patient, successful_channels))
        else:
            skipped_patients.append(patient)

    except Exception as e:
        skipped_patients.append(patient)


X = np.array(X)
y = np.array(y)


print("ECG MULTIFRACTAL CLASSIFICATION\n")
print(f"Dataset shape: {X.shape} (samples, features)")
print(f"Channels used: V1-V6 (6 chest electrodes)")
print(f"Features per subject: {X.shape[1]}")
label_counts = Counter(y.tolist())
print(f"Label distribution: {dict(label_counts)}")
print(f"  - Healthy (0): {label_counts[0]} patients")
print(f"  - Pathological (1): {label_counts[1]} patients")

print(f"\nData Quality Report:")
print(f"  - Skipped patients (insufficient data): {len(skipped_patients)}")
print(f"  - Patients with partial channels (< 6): {len(partial_channels)}")
if partial_channels:
    print(f"    Examples: {partial_channels[:5]}")
print(f"  - Mean features per subject: {X.shape[1]:.1f} (should be ~30)")
print("="*70 + "\n")

# Run stratified k-fold cross-validation
train_svc_cv(X, y, n_splits=5)
