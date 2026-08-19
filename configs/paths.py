from pathlib import Path

# All paths here are relative to the project root. Each caller is
# responsible for resolving its own PROJECT_ROOT (via its own __file__)
# and joining it with these, since how many parents you need to reach
# root depends on how deep the calling file sits.

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_PICKLE_DIR = DATA_DIR / "processed" / "parsed"
SAMPLED_PICKLE_DIR = DATA_DIR / "processed" / "sampled"

PAN12_TRAINING_XML_PATH = RAW_DATA_DIR / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PAN12_TEST_XML_PATH = RAW_DATA_DIR / "pan12-test" / "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"

PARSED_TRAINING_PICKLE_PATH = PARSED_PICKLE_DIR / "training_df.pkl"
PARSED_TEST_PICKLE_PATH = PARSED_PICKLE_DIR / "test_df.pkl"

SAMPLED_TRAIN_PICKLE_PATH = SAMPLED_PICKLE_DIR / "train_split_df.pkl"
SAMPLED_TEST_PICKLE_PATH = SAMPLED_PICKLE_DIR / "test_split_df.pkl"

TRAINING_PREDATOR_IDS_PATH = RAW_DATA_DIR / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
TEST_PREDATOR_IDS_PATH = RAW_DATA_DIR / "pan12-test" / "pan12-sexual-predator-identification-groundtruth-problem1.txt"
TEST_SUSPICIOUS_LINES_PATH = RAW_DATA_DIR / "pan12-test" / "pan12-sexual-predator-identification-groundtruth-problem2.txt"
