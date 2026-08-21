const BAND_LABEL = { High: "⚠️ High", Medium: "▲ Medium", Low: "✓ Low" };

function collectConversationText() {
  const bubbles = document.querySelectorAll("#chat-log .chat-message");
  const lines = [];
  bubbles.forEach((el) => {
    const speaker = el.dataset.sender;
    const text = el.querySelector(".message-text").textContent;
    lines.push(`${speaker}: ${text}`);
  });
  return lines.join("\n");
}

async function scanAndBadge() {
  const rawText = collectConversationText();
  if (!rawText) return;

  const response = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText }),
  });
  const data = await response.json();

  const bubbles = document.querySelectorAll("#chat-log .chat-message");
  bubbles.forEach((el, i) => {
    const existingBadge = el.querySelector(".garda-badge");
    if (existingBadge) existingBadge.remove();

    const result = data.results[i];
    const badge = document.createElement("span");
    badge.className = `garda-badge garda-risk-${result.risk_band.toLowerCase()}`;
    badge.textContent = BAND_LABEL[result.risk_band];
    el.appendChild(badge);
  });

  chrome.storage.local.set({ "garda:summary": data.summary });
}

document.addEventListener("DOMContentLoaded", () => {
  scanAndBadge();

  // The mock page has no knowledge of the extension - we attach our own
  // listener to its send button rather than expecting it to notify us.
  const sendBtn = document.getElementById("send-btn");
  if (sendBtn) sendBtn.addEventListener("click", () => setTimeout(scanAndBadge, 50));
});
