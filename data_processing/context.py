# Builds windowed conversation context around each message, so the
# classifier sees the lead-up to a line instead of judging it in isolation.

def build_context_window(messages, authors, window=3):
    """
    Given ordered messages and authors from a single conversation, returns
    one context string per message: the previous `window` messages plus
    the message itself, each tagged "self"
    """
    context_texts = []
    for i in range(len(messages)):
        start = max(0, i - window)
        lines = [
            f"{'self' if authors[j] == authors[i] else 'other'}: {messages[j]}"
            for j in range(start, i + 1)
        ]
        context_texts.append("\n".join(lines))
    return context_texts

def build_context_text(df, window=3):
    """
    Adds a "context_text" column to df by running build_context_window
    over each conversation_id group, ordered by line.
    """
    df = df.copy()
    df["line"] = df["line"].astype(int)
    df = df.sort_values(["conversation_id", "line"]).reset_index(drop=True)

    context_texts = []
    for _, group in df.groupby("conversation_id", sort=False):
        context_texts.extend(build_context_window(group["text"].tolist(), group["author_id"].tolist(), window=window))

    df["context_text"] = context_texts
    return df
