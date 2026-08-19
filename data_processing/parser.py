# Parses the PAN12 XML Files
from pathlib import Path
from lxml import etree
import pandas as pd

from configs import paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAN12_TRAINING_PATH = PROJECT_ROOT / paths.PAN12_TRAINING_XML_PATH
PAN12_TEST_PATH = PROJECT_ROOT / paths.PAN12_TEST_XML_PATH
TRAINING_PICKLE_PATH = PROJECT_ROOT / paths.PARSED_TRAINING_PICKLE_PATH
TEST_PICKLE_PATH = PROJECT_ROOT / paths.PARSED_TEST_PICKLE_PATH

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