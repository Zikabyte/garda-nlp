const BACKEND_URL = "http://localhost:8000";

function showStatus() {
  chrome.storage.local.get("garda:summary", (data) => {
    const status = document.getElementById("status");
    status.textContent = data["garda:summary"] || "No analysis yet - open the mock chat page.";
  });
}

function renderResults(data) {
  document.getElementById("summary").textContent = data.summary;

  const container = document.getElementById("results");
  container.innerHTML = "";
  data.results.forEach((r) => {
    const row = document.createElement("div");
    row.className = `result-row risk-${r.risk_band.toLowerCase()}`;
    row.textContent = `[${r.speaker}] ${r.text} - ${(r.risk_score * 100).toFixed(1)}% (${r.risk_band})`;
    container.appendChild(row);
  });
}

async function analyze() {
  const rawText = document.getElementById("input").value;
  const summaryEl = document.getElementById("summary");

  try {
    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawText }),
    });
    if (!response.ok) throw new Error("Backend error");
    renderResults(await response.json());
  } catch (err) {
    summaryEl.textContent = "Backend not running - start it with: python -m extension.backend.main";
    document.getElementById("results").innerHTML = "";
  }
}

// Runs inside the target tab (not the popup) - reads the page's own
// chat-log DOM, same shape content.js expects/produces.
function extractConversationFromPage() {
  const bubbles = document.querySelectorAll("#chat-log .chat-message");
  if (!bubbles.length) return null;

  const lines = [];
  bubbles.forEach((el) => {
    const speaker = el.dataset.sender;
    const text = el.querySelector(".message-text").textContent;
    lines.push(`${speaker}: ${text}`);
  });
  return lines.join("\n");
}

async function loadFromCurrentPage() {
  const summaryEl = document.getElementById("summary");
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractConversationFromPage,
    });

    if (!result) {
      summaryEl.textContent = "No chat found on this page - open the mock chat page first.";
      return;
    }

    document.getElementById("input").value = result;
    analyze();
  } catch (err) {
    summaryEl.textContent = "Couldn't read this page (try reloading it first).";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  showStatus();
  document.getElementById("analyze-btn").addEventListener("click", analyze);
  document.getElementById("load-page-btn").addEventListener("click", loadFromCurrentPage);
});
