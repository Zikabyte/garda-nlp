# Parses the PAN12 XML Files
from pathlib import Path
from lxml import etree
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PAN12_TRAINING_PATH = RAW_DATA_DIR / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PAN12_TEST_PATH = RAW_DATA_DIR / "pan12-test" / "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"

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

def parse_pan12_training():
    return _parse_pan12_xml(PAN12_TRAINING_PATH)

def parse_pan12_test():
    return _parse_pan12_xml(PAN12_TEST_PATH)

def main():
    training_df = parse_pan12_training()
    test_df = parse_pan12_test()

    # Pickle
    training_df.to_pickle(DATA_DIR / "processed" / "pickle" / "training_rows.pkl")
    test_df.to_pickle(DATA_DIR / "processed" / "pickle" / "test_rows.pkl")

    # Display Information
    print(f"Conversation Count (Training): {training_df['conversation_id'].nunique()}")
    print(f"Conversation Count (Test): {test_df['conversation_id'].nunique()}")

if __name__ == '__main__':
    main()