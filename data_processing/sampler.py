import pandas as pd

from labeler import (
    TRAINING_PICKLE_DIR,
    TRAINING_PREDATOR_IDS_DIR,
    load_df,
    load_predator_ids,
    label_by_authors,
)

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
    training_df = load_df(TRAINING_PICKLE_DIR)
    training_predator_ids = load_predator_ids(TRAINING_PREDATOR_IDS_DIR)
    labeled_df = label_by_authors(training_df, training_predator_ids)

    print("[Before Sampling]\n", labeled_df["is_suspicious"].value_counts(), end="\n\n")

    sampled_df = sample_balanced(labeled_df)

    print("[After Sampling]\n", sampled_df["is_suspicious"].value_counts())

if __name__ == "__main__":
    main()
