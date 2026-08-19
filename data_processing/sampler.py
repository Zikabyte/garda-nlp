import pandas as pd
from sklearn.model_selection import train_test_split

from labeler import (
    TEST_PICKLE_DIR,
    TEST_SUSPICIOUS_LINES_DIR,
    load_df,
    load_suspicious_lines,
    label_by_lines,
)

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

def sample_balanced(df, label_col="is_suspicious", ratio=1.0, random_state=42):
    """
    Returns a shuffled sample containing every positive row plus a random
    sample of negative rows, sized len(positives) * ratio, so the minority
    class isn't drowned out during training.
    """
    positives = df[df[label_col]]
    negatives = df[~df[label_col]]

    negative_sample = negatives.sample(n=int(len(positives) * ratio), random_state=random_state)

    sampled = pd.concat([positives, negative_sample])
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

def main():
    # PAN12's "test" corpus is our only source of line-level (so that it's precise)
    corpus_df = load_df(TEST_PICKLE_DIR)
    suspicious_lines = load_suspicious_lines(TEST_SUSPICIOUS_LINES_DIR)
    labeled_df = label_by_lines(corpus_df, suspicious_lines)

    train_df, test_df = train_test_split_df(labeled_df)

    print("[Train Split]\n", train_df["is_suspicious"].value_counts(), end="\n\n")
    print("[Test Split]\n", test_df["is_suspicious"].value_counts(), end="\n\n")

    balanced_train_df = sample_balanced(train_df)

    print("[Balanced Train Split]\n", balanced_train_df["is_suspicious"].value_counts())

if __name__ == "__main__":
    main()
