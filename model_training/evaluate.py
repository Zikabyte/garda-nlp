from setfit import SetFitModel
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from configs import paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / paths.SETFIT_MODEL_PATH
TEST_PICKLE_PATH = PROJECT_ROOT / paths.SAMPLED_TEST_PICKLE_PATH

def find_threshold(y_true, probas, min_recall=0.8):
    """
    Finds the decision threshold with the best precision among those that
    still keep recall on the suspicious class at or above min_recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
    precisions, recalls = precisions[:-1], recalls[:-1]

    candidates = recalls >= min_recall
    if not candidates.any():
        return 0.5

    best_idx = precisions[candidates].argmax()
    return thresholds[candidates][best_idx]

def evaluate(y_true, predictions, label=""):
    """
    Prints precision/recall/F1 per class plus the confusion matrix. Recall
    on the suspicious class matters most here
    """
    if label:
        print(f"--- {label} ---")
    print(classification_report(y_true, predictions))
    print(confusion_matrix(y_true, predictions))

def print_samples(model, test_df, n=10, random_state=42):
    """
    Prints n rows with their true label and predicted probability, sampled
    evenly between classes (n // 2 each) rather than purely at random
    """
    per_class = n // 2
    positives = test_df[test_df["label"] == 1].sample(n=per_class, random_state=random_state)
    negatives = test_df[test_df["label"] == 0].sample(n=per_class, random_state=random_state)

    sample = pd.concat([positives, negatives]).sample(frac=1, random_state=random_state)
    probas = model.predict_proba(sample["context_text"].tolist(), as_numpy=True)[:, 1]

    for text, true_label, proba in zip(sample["text"], sample["label"], probas):
        print(f"[label={true_label}] [proba={proba:.4f}] {text}")

def main():
    # Load model
    model = SetFitModel.from_pretrained(MODEL_SAVE_PATH)

    # Load test dataset
    test_df = pd.read_pickle(TEST_PICKLE_PATH)

    # Same rename/cast as train.py, so predictions line up against "label"
    test_df = test_df.rename(columns={"is_suspicious": "label"})
    test_df["label"] = test_df["label"].astype(int)

    probas = model.predict_proba(test_df["context_text"].tolist(), as_numpy=True)[:, 1]

    fixed_predictions = (probas >= 0.95).astype(int)
    evaluate(test_df["label"], fixed_predictions, label="Fixed threshold (0.95)")

if __name__ == "__main__":
    main()
