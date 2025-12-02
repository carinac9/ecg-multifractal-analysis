from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def train_svc(X, y, test_size=0.25):
    # require at least two classes to train an SVC
    unique_classes = set(y.tolist()) if hasattr(y, 'tolist') else set(y)
    if len(unique_classes) < 2:
        print("train_svc: need at least two classes to train SVC; found classes=", unique_classes)
        return None, None, None

    model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # double-check training split has multiple classes
    if len(set(y_train.tolist())) < 2:
        print("train_svc: training split contains a single class after stratify; aborting training")
        return None, None, None

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return model, acc, cm
