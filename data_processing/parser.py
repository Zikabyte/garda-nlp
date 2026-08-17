# Parses the PAN12 XML Files
from pathlib import Path
from lxml import etree
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PICKLE_DIR = DATA_DIR / "processed" / "pickle"
PAN12_TRAINING_PATH = RAW_DATA_DIR / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PAN12_TEST_PATH = RAW_DATA_DIR / "pan12-test" / "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
TRAINING_PICKLE_PATH = PICKLE_DIR / "training_df.pkl"
TEST_PICKLE_PATH = PICKLE_DIR / "test_df.pkl"

def _parse_pan12_xml(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    rows = []
    for conversation in root.findall("conversation"):
        conv_id = conversation.get("id")
        for message in conversation.findall("message"):
            rows.append({
                "conversation_id": conv_id,
                "author_id": message.findtext("author"),
                "time": message.findtext("time"),
                "text": message.findtext("text"),
                "line": message.get("line")
            })

    return pd.DataFrame(rows)

def _load_or_parse(xml_path, pickle_path, force=False):
    if not force and pickle_path.exists():
        return pd.read_pickle(pickle_path)

    df = _parse_pan12_xml(xml_path)
    df.to_pickle(pickle_path)
    return df

def parse_pan12_training(force=False):
    return _load_or_parse(PAN12_TRAINING_PATH, TRAINING_PICKLE_PATH, force=force)

def parse_pan12_test(force=False):
    return _load_or_parse(PAN12_TEST_PATH, TEST_PICKLE_PATH, force=force)

def main():
    training_df = parse_pan12_training()
    test_df = parse_pan12_test()

    # Display Information
    print(f"Conversation Count (Training): {training_df['conversation_id'].nunique()}")
    print(f"Conversation Count (Test): {test_df['conversation_id'].nunique()}")

if __name__ == '__main__':
    main()