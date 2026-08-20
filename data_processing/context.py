# Builds windowed conversation context around each message, so the
# classifier sees the lead-up to a line instead of judging it in isolation.

def build_context_text(df, window=3):
    """
    Adds a "context_text" column: the target message prefixed by the
    previous `window` messages from the same conversation (in order).
    Each line is tagged "self" (same author as the target message) or
    "other" (a different author) instead of the raw author_id, so the
    model sees turn-taking structure without depending on specific
    participant identities - useful both for training and for scoring
    conversations pasted into the demo later, where author ids won't
    match anything the model has seen.
    """
    df = df.copy()
    df["line"] = df["line"].astype(int)
    df = df.sort_values(["conversation_id", "line"]).reset_index(drop=True)

    context_texts = []
    for _, group in df.groupby("conversation_id", sort=False):
        messages = group["text"].tolist()
        authors = group["author_id"].tolist()

        for i in range(len(messages)):
            start = max(0, i - window)
            lines = [
                f"{'self' if authors[j] == authors[i] else 'other'}: {messages[j]}"
                for j in range(start, i + 1)
            ]
            context_texts.append("\n".join(lines))

    df["context_text"] = context_texts
    return df
