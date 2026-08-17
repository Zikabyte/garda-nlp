# Parses the PAN12 XML Files
from pathlib import Path
from lxml import etree
import pickle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PAN12_TRAINING_PATH = RAW_DATA_DIR / "pan12-training" / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PAN12_TEST_PATH = RAW_DATA_DIR / "pan12-test" / "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"

def parse_pan12_training():
    tree = etree.parse(PAN12_TRAINING_PATH)
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

    return (rows, root)

def parse_pan12_test():
    tree = etree.parse(PAN12_TEST_PATH)
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
                
    return (rows, root)

def main():
    training_rows, training_root = parse_pan12_training()
    test_rows, test_root = parse_pan12_test()

    # Pickle
    with open(DATA_DIR / "processed" / "pickle" / "training_rows.pkl", "wb") as file:
        pickle.dump(training_rows, file)

    with open(DATA_DIR / "processed" / "pickle" / "test_rows.pkl", "wb") as file:
         pickle.dump(test_rows, file)

    # Display Information 
    print(f"Conversation Count (Training): {len(training_root)}")
    print(f"Conversation Count (Test): {len(test_root)}")

if __name__ == '__main__':
    main()