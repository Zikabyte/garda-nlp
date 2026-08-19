from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_PICKLE_DIR = DATA_DIR / "processed" / "pickle" / "training_df.pkl"
TRAINING_PREDATOR_IDS_DIR = DATA_DIR / "raw" / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"

TEST_PICKLE_DIR = DATA_DIR / "processed" / "pickle" / "test_df.pkl"
TEST_PREDATOR_IDS_DIR = DATA_DIR / "raw" / "pan12-test" / "pan12-sexual-predator-identification-groundtruth-problem1.txt"
TEST_SUSPICIOUS_LINES_DIR = DATA_DIR / "raw" / "pan12-test" / "pan12-sexual-predator-identification-groundtruth-problem2.txt"

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

def load_suspicious_lines(path):
    """
    Returns conversation_id + line pairs considered suspicious
    """
    return pd.read_csv(path, sep="\t", names=["conversation_id", "line"], dtype=str)

def label_by_lines(df, suspicious_lines_df):
    suspicious_lines_df = suspicious_lines_df.copy()
    suspicious_lines_df["is_suspicious"] = True

    merged = df.merge(suspicious_lines_df, on=["conversation_id", "line"], how="left")
    merged["is_suspicious"] = merged["is_suspicious"].fillna(False).astype(bool)

    return merged

def main():
    # Labeling by Author
    training_df = load_df(TRAINING_PICKLE_DIR)
    training_predator_ids = load_predator_ids(TRAINING_PREDATOR_IDS_DIR)

    test_df = load_df(TEST_PICKLE_DIR)
    test_predator_ids = load_predator_ids(TEST_PREDATOR_IDS_DIR)

    # Labeling by Lines 
    test_df_by_lines = load_df(TEST_PICKLE_DIR)
    test_suspicious_lines = load_suspicious_lines(TEST_SUSPICIOUS_LINES_DIR)

    # Display Information
    print("[Training Dataframe]\n", label_by_authors(training_df, training_predator_ids), end="\n\n")
    print("[Test Dataframe (by author)]\n", label_by_authors(test_df, test_predator_ids), end="\n\n")
    print("[Test Dataframe (by line)]\n", label_by_lines(test_df_by_lines, test_suspicious_lines))

if __name__ == "__main__":
    main()