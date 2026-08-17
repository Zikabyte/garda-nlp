from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_PICKLE_DIR = DATA_DIR / "processed" / "pickle" / "training_df.pkl"
TRAINING_PREDATOR_IDS_DIR = DATA_DIR / "raw" / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"

def load_df(path):
    """
    Returns training_rows from pickle in the form of a dataframe
    """
    training_df = pd.read_pickle(path)
    return training_df

def load_predator_ids(path):
    """
    Returns predator ids from both the training dataset
    """
    ids = set()

    with open(path, "r") as file:
        for line in file:
            ids.add(line.strip())

    return ids

def label_by_authors(df, predator_ids):
    id_series = df["author_id"]
    df["is_suspicious"] = id_series.isin(predator_ids)

    return df

def main():
    training_df = load_df(TRAINING_PICKLE_DIR)
    training_predator_ids = load_predator_ids(TRAINING_PREDATOR_IDS_DIR)

    # Display Information
    print("[Training Dataframe]\n", label_by_authors(training_df, training_predator_ids))

if __name__ == "__main__":
    main()