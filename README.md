# ECG Multifractal Classification

A Python project for extracting multifractal features from ECG signals (PTB-DB dataset) and training an SVM classifier to distinguish between healthy and pathological records.

## Overview

This project implements a complete pipeline:
1. **Preprocessing**: Filter and normalize raw ECG signals
2. **Delay embedding**: Embed 1D time series into high-dimensional attractors (M=4, τ=198)
3. **Multifractal analysis**: Compute multifractal spectrum (α, f(α)) using partition function method
4. **Feature extraction**: Extract 6 engineered features from the spectrum
5. **Classification**: Train an SVM with RBF kernel to classify healthy vs pathological

## Dataset

The project uses the **PTB-DB** (PhysioNet PTB Database) of ECG recordings. 
- Contains recordings from ~300 patients (healthy controls and various cardiac pathologies)
- Each record includes multiple leads; this project uses lead 6
- Access: https://physionet.org/content/ptb/1.0.0/

**Note**: Data is not included in the repo due to size. Download separately and place in `data/ptbdb/`.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/ecg_multifractal_project.git
cd ecg_multifractal_project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline (requires data in data/ptbdb/)
python main.py
```

The script will:
1. Load all records from `RECORDS` file
2. Extract features for each unique patient
3. Assign labels based on `CONTROLS` file (0=healthy, 1=pathological)
4. Train an SVM and report accuracy and confusion matrix

### Output
```
Extracted dataset shape: (N_samples, 6)
Label distribution: Counter({...})

Accuracy: 0.XX
Confusion matrix:
 [[TP FN]
  [FP TN]]
```

## Project Structure

```
ecg_multifractal_project/
├── main.py                 # Main pipeline
├── requirements.txt        # Dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── src/
    ├── preprocessing.py   # ECG filtering and normalization
    ├── embedding.py       # Delay embedding & tau estimation
    ├── multifractal.py    # Multifractal spectrum calculation
    ├── features.py        # Feature extraction from spectrum
    └── classification.py  # SVM training and evaluation
```

## Key Functions

### `src/multifractal.py`
- `subsample(Y, n=5000)`: Uniformly subsample embedded attractor points
- `multifractal_spectrum(Y, q_vals, num_boxes=50)`: Compute α and f(α) via partition function method

### `src/features.py`
- `extract_alpha0(alpha, f)`: Extract α₀ (Hurst exponent)
- `extract_alpha_edges(alpha, f)`: Extract α₁, α₂ (spectrum width)
- `compute_slopes(alpha, f)`: Extract left and right slopes of spectrum

### `src/classification.py`
- `train_svc(X, y, test_size=0.25)`: Train SVM with stratified 75-25 train-test split

## Configuration

Edit `main.py` to adjust:
- `DATA_DIR`: Path to PTB-DB directory
- `M`: Embedding dimension (default: 4)
- `TAU`: Time delay for embedding (default: 198 samples)
- `Q_VALS`: Range of q-values for multifractal spectrum (default: [-5, 5])

## Error Handling

The script includes guards against:
- Single-class label sets (classification is skipped with a message)
- Degenerate or empty time series (safe fallback spectrum returned)
- Missing channels or files (records are skipped with error logging)

## Notes

- **Labeling**: Labels are assigned per-patient (0=healthy if patient in `CONTROLS`, 1=pathological otherwise)
- **Reproducibility**: Random seed fixed in `train_test_split` (random_state=42)
- **Class imbalance**: SVM uses `class_weight='balanced'` to handle unequal class sizes

## Future Improvements

- Add unit tests for edge cases
- Implement deterministic seed for all stochastic steps (preprocessing, embedding)
- Add per-record labeling option
- Integrate CI/CD (GitHub Actions)
- Parallelize feature extraction using joblib

## License

This project is provided as-is for educational and research purposes. 
The PTB-DB dataset is governed by its own license; see https://physionet.org/content/ptb/1.0.0/

## Contact

For questions or issues, open an issue on GitHub.
