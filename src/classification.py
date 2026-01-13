from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np


def train_svc(X, y, test_size=0.2):
    unique_classes = set(y.tolist()) if hasattr(y, 'tolist') else set(y)
    if len(unique_classes) < 2:
        print("train_svc: need at least two classes to train SVC; found classes=", unique_classes)
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    if len(set(y_train.tolist())) < 2:
        print("train_svc: training split contains a single class after stratify; aborting training")
        return None, None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if X_train_scaled.shape[1] > 20:
        pca = PCA(n_components=min(20, X_train_scaled.shape[0] // 2))
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(
            f"PCA: Reduced to {pca.n_components} dimensions (explained variance: {sum(pca.explained_variance_ratio_):.2%})")

    print("\n Training Individual Models ")

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=1)
    rf.fit(X_train_scaled, y_train)
    preds_rf = rf.predict(X_test_scaled)
    preds_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf = f1_score(y_test, preds_rf)
    print(f"RF - Accuracy: {acc_rf:.4f}, F1: {f1_rf:.4f}")

    print("\n Training SVM ")
    param_grid = {'C': [1, 10, 100], 'kernel': [
        'rbf', 'linear'], 'gamma': ['scale', 0.01]}

    svc = SVC(class_weight='balanced', random_state=42, probability=True)
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring='f1', n_jobs=1)
    grid_search.fit(X_train_scaled, y_train)

    print(
        f"Best SVM params: {grid_search.best_params_}, F1: {grid_search.best_score_:.4f}")

    model_svm = grid_search.best_estimator_
    preds_svm = model_svm.predict(X_test_scaled)
    preds_proba_svm = model_svm.predict_proba(X_test_scaled)[:, 1]
    acc_svm = accuracy_score(y_test, preds_svm)
    f1_svm = f1_score(y_test, preds_svm)
    print(f"SVM - Accuracy: {acc_svm:.4f}, F1: {f1_svm:.4f}")

    print("\n Training Ensemble Classifier ")
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('svm', model_svm)],
        voting='soft',
        weights=[1, 1]
    )
    voting_clf.fit(X_train_scaled, y_train)
    preds_ensemble = voting_clf.predict(X_test_scaled)
    preds_proba_ensemble = voting_clf.predict_proba(X_test_scaled)[:, 1]
    acc_ensemble = accuracy_score(y_test, preds_ensemble)
    f1_ensemble = f1_score(y_test, preds_ensemble)
    print(f"Ensemble - Accuracy: {acc_ensemble:.4f}, F1: {f1_ensemble:.4f}")

    accuracies = {'RF': acc_rf, 'SVM': acc_svm, 'Ensemble': acc_ensemble}
    best_name = max(accuracies, key=accuracies.get)
    print(f"\n✓ Best model: {best_name}")

    if best_name == 'Ensemble':
        model = voting_clf
        preds = preds_ensemble
        preds_proba = preds_proba_ensemble
        acc = acc_ensemble
    elif best_name == 'RF':
        model = rf
        preds = preds_rf
        preds_proba = preds_proba_rf
        acc = acc_rf
    else:
        model = model_svm
        preds = preds_svm
        preds_proba = preds_proba_svm
        acc = acc_svm

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, preds_proba)
    cm = confusion_matrix(y_test, preds)

    print(f"\n FINAL MODEL PERFORMANCE ")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return model, acc, cm


def train_svc_cv(X, y, n_splits=5):
    print(f"Running Stratified {n_splits}-Fold Cross-Validation...")
    print("=" * 80 + "\n")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pipeline: standardize → PCA → Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=20, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100,
         random_state=42, class_weight='balanced', n_jobs=-1))
    ])

    # Define metrics to track
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    # Run cross-validation
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)

    # Print results
    print("CROSS-VALIDATION RESULTS")
    print(f"{'Metric':<15} {'Test (mean ± std)':<30} {'Train (mean ± std)':<30}")

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        print(f"{metric.upper():<15} {test_scores.mean():.4f} ± {test_scores.std():.4f}          {train_scores.mean():.4f} ± {train_scores.std():.4f}")

    print("\nKey Observations:")
    print(
        f"  • Sensitivity (Recall):  {cv_results['test_recall'].mean():.2%} → Ability to detect pathology")
    print(
        f"  • Precision:             {cv_results['test_precision'].mean():.2%} → Avoid false alarms")
    print(
        f"  • F1-Score:              {cv_results['test_f1'].mean():.2%} → Balanced performance metric")
    print(
        f"  • ROC-AUC:               {cv_results['test_roc_auc'].mean():.4f} → Discrimination across thresholds")

    train_acc = cv_results['train_accuracy'].mean()
    test_acc = cv_results['test_accuracy'].mean()
    gap = train_acc - test_acc

    print(f"\nOverfitting Check:")
    print(
        f"  • Train-Test Gap:        {gap:.4f} (good if < 0.1, overfitting if > 0.2)")

    if gap < 0.05:
        print("  Model generalizes well!")
    elif gap < 0.15:
        print("  Slight overfitting detected")
    else:
        print("  Significant overfitting")

    print("\n")
