import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from configs import paths
from data_processing.labeler import (
    TEST_PICKLE_DIR,
    TEST_SUSPICIOUS_LINES_DIR,
    load_df,
    load_suspicious_lines,
)
from data_processing.sampler import load_train_test_df

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / paths.SETFIT_MODEL_PATH

def train(train_dataset):
    # Load pretrained model
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Setup trainer
    args = TrainingArguments(batch_size=16, num_epochs=1)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)

    # Train
    trainer.train()

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)

def main():
    corpus_df = load_df(TEST_PICKLE_DIR)
    suspicious_lines = load_suspicious_lines(TEST_SUSPICIOUS_LINES_DIR)
    train_df, test_df = load_train_test_df(corpus_df, suspicious_lines, max_per_class=500, ratio=5)

    # SetFit expects a "label" column, and an int rather than bool
    train_df = train_df.rename(columns={"is_suspicious": "label"})
    train_df["label"] = train_df["label"].astype(int)

    # SetFit's Trainer expects a "text" column; our source column is
    # "context_text" (message + preceding conversation window)
    train_dataset = Dataset.from_pandas(train_df[["context_text", "label"]].rename(columns={"context_text": "text"}))

    train(train_dataset)

if __name__ == "__main__":
    main()
