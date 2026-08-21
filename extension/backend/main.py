from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from setfit import SetFitModel

from configs import paths
from data_processing.context import build_context_window

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / paths.SETFIT_MODEL_PATH
MOCK_PAGE_DIR = PROJECT_ROOT / "extension" / "mock_page"
CONTEXT_WINDOW = 3

model = SetFitModel.from_pretrained(MODEL_SAVE_PATH)

def parse_conversation(raw_text):
    """
    Parses "Speaker: message" lines into ordered authors/messages lists,
    skipping blank lines. A line without a ":" continues the previous
    speaker. Same convention as demo/app.py.
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

class AnalyzeRequest(BaseModel):
    raw_text: str

class MessageResult(BaseModel):
    speaker: str
    text: str
    risk_score: float
    risk_band: str

class AnalyzeResponse(BaseModel):
    results: list[MessageResult]
    summary: str

app = FastAPI(title="GARDA Extension Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    authors, messages = parse_conversation(request.raw_text)

    if not messages:
        return AnalyzeResponse(results=[], summary="Paste a conversation first.")

    context_texts = build_context_window(messages, authors, window=CONTEXT_WINDOW)
    probas = model.predict_proba(context_texts, as_numpy=True)[:, 1]

    results = [
        MessageResult(speaker=author, text=message, risk_score=float(proba), risk_band=risk_band(proba))
        for author, message, proba in zip(authors, messages, probas)
    ]

    top_idx = probas.argmax()
    summary = (
        f"Highest-risk line: [{authors[top_idx]}] \"{messages[top_idx]}\" "
        f"- {probas[top_idx]:.1%} ({risk_band(probas[top_idx])} risk)"
    )

    return AnalyzeResponse(results=results, summary=summary)

app.mount("/mock-chat", StaticFiles(directory=MOCK_PAGE_DIR, html=True), name="mock_chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
