from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from configs import paths
from data_processing.labeler import (
    TEST_PICKLE_DIR,
    TEST_SUSPICIOUS_LINES_DIR,
    load_df,
    load_suspicious_lines,
    label_by_lines,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLED_TRAIN_PICKLE_PATH = PROJECT_ROOT / paths.SAMPLED_TRAIN_PICKLE_PATH
SAMPLED_TEST_PICKLE_PATH = PROJECT_ROOT / paths.SAMPLED_TEST_PICKLE_PATH

def train_test_split_df(df, test_size=0.2, random_state=42):
    """
    Splits df into train/test by conversation_id (not by row), so messages
    from the same conversation never end up on both sides. A random per-row
    split would leak conversation/author style across the split and inflate
    eval scores artificially.
    """
    conversation_ids = df["conversation_id"].unique().to_numpy()
    train_ids, test_ids = train_test_split(conversation_ids, test_size=test_size, random_state=random_state)

    train_df = df[df["conversation_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["conversation_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, test_df

def sample_balanced(df, label_col="is_suspicious", ratio=1.0, max_per_class=None, random_state=42):
    """
    Returns a shuffled sample containing every positive row plus a random
    sample of negative rows, sized len(positives) * ratio, so the minority
    class isn't drowned out during training.

    max_per_class caps how many positive rows are used. SetFit is a few-shot 
    method -> it's meant to learn from tens to a few hundred examples per class, 
    """
    positives = df[df[label_col]]
    negatives = df[~df[label_col]]

    if max_per_class is not None and len(positives) > max_per_class:
        positives = positives.sample(n=max_per_class, random_state=random_state)

    negative_sample = negatives.sample(n=int(len(positives) * ratio), random_state=random_state)

    sampled = pd.concat([positives, negative_sample])
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

def load_train_test_df(corpus_df, suspicious_lines, force=False, max_per_class=None):
    """
    Loads balanced train dataframe and test dataframe
    from pickle (if exists) or newly generated
    """
    if not force and SAMPLED_TRAIN_PICKLE_PATH.exists() and SAMPLED_TEST_PICKLE_PATH.exists():
        return pd.read_pickle(SAMPLED_TRAIN_PICKLE_PATH), pd.read_pickle(SAMPLED_TEST_PICKLE_PATH)

    labeled_df = label_by_lines(corpus_df, suspicious_lines)
    train_df, test_df = train_test_split_df(labeled_df)
    balanced_train_df = sample_balanced(train_df, max_per_class=max_per_class)

    SAMPLED_TRAIN_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced_train_df.to_pickle(SAMPLED_TRAIN_PICKLE_PATH)
    test_df.to_pickle(SAMPLED_TEST_PICKLE_PATH)

    return balanced_train_df, test_df

def main():
    # PAN12's "test" corpus is our only source of line-level (so that it's precise)
    corpus_df = load_df(TEST_PICKLE_DIR)
    suspicious_lines = load_suspicious_lines(TEST_SUSPICIOUS_LINES_DIR)

    balanced_train_df, test_df = load_train_test_df(corpus_df, suspicious_lines)

    print("[Balanced Train Split]\n", balanced_train_df["is_suspicious"].value_counts(), end="\n\n")
    print("[Test Split]\n", test_df["is_suspicious"].value_counts())


if __name__ == "__main__":
    main()
