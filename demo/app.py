import random
from pathlib import Path

import gradio as gr
import pandas as pd
from setfit import SetFitModel

from configs import paths
from data_processing.context import build_context_window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / paths.SETFIT_MODEL_PATH
CONTEXT_WINDOW = 3

EXAMPLE_CONVERSATIONS = [
    """A: hey whats up
B: nm just chilling, u?
A: same. hows school
B: its ok i guess
A: ur cute btw, how old are you?
B: 13 lol
A: nice, wanna send me a pic""",
    """A: yo did you finish the homework
B: not yet lol, math is killing me
A: same, question 4 makes no sense
B: right?? gonna ask the teacher tmrw
A: good idea, im just gonna wing it""",
    """A: hi, been a while since we talked
B: yeah! how have you been
A: good good, you seem really mature for your age btw
B: haha thanks i guess
A: whats your address, i wanna send you something""",
]

model = SetFitModel.from_pretrained(MODEL_SAVE_PATH)

def parse_conversation(raw_text):
    """
    Parses pasted "Speaker: message" lines into ordered authors/messages
    lists, skipping blank lines. A line without a ":" is treated as a
    continuation from the same speaker as the previous line.
    """
    authors, messages = [], []
    last_author = None

    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        if ":" in line:
            author, message = line.split(":", 1)
            author, message = author.strip(), message.strip()
        else:
            author, message = last_author or "unknown", line

        authors.append(author)
        messages.append(message)
        last_author = author

    return authors, messages

def risk_band(proba):
    if proba >= 0.95:
        return "High"
    if proba >= 0.5:
        return "Medium"
    return "Low"

def analyze(raw_text):
    authors, messages = parse_conversation(raw_text)

    empty_table = pd.DataFrame(columns=["Speaker", "Message", "Risk Score", "Risk Band"])
    if not messages:
        return empty_table, "Paste a conversation first."

    context_texts = build_context_window(messages, authors, window=CONTEXT_WINDOW)
    probas = model.predict_proba(context_texts, as_numpy=True)[:, 1]

    table = pd.DataFrame({
        "Speaker": authors,
        "Message": messages,
        "Risk Score": [f"{p:.1%}" for p in probas],
        "Risk Band": [risk_band(p) for p in probas],
    })

    top_idx = probas.argmax()
    summary = (
        f"Highest-risk line: [{authors[top_idx]}] \"{messages[top_idx]}\" "
        f"- {probas[top_idx]:.1%} ({risk_band(probas[top_idx])} risk)"
    )

    return table, summary

def load_example():
    choice = random.choice(EXAMPLE_CONVERSATIONS)
    gr.Info("Example Loaded!")
    return choice

with gr.Blocks(title="GARDA - Grooming Alert and Detection Analysis") as demo:
    gr.Markdown("# GARDA - Grooming Alert and Detection Analysis (Demo)")
    gr.Markdown(
        "Paste a chat conversation below, one message per line as `Speaker: message`. "
        "Each line is scored using the model's fine-tuned SetFit classifier, using the "
        "few messages before it as context."
    )

    input_box = gr.Textbox(
        lines=10,
        label="Conversation",
        placeholder="A: hey\nB: hi\nA: how old are you?",
    )
    with gr.Row():
        analyze_btn = gr.Button("Analyze")
        example_btn = gr.Button("Load Example")
    summary_box = gr.Textbox(label="Summary", interactive=False)
    output_table = gr.Dataframe(
        headers=["Speaker", "Message", "Risk Score", "Risk Band"],
        label="Per-line risk",
    )

    analyze_btn.click(fn=analyze, inputs=input_box, outputs=[output_table, summary_box])
    example_btn.click(fn=load_example, outputs=input_box)

if __name__ == "__main__":
    demo.launch()
