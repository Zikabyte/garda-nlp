// Page-owned chat logic only - no knowledge of the extension/backend.
document.getElementById("send-btn").addEventListener("click", () => {
  const input = document.getElementById("msg-input");
  const text = input.value.trim();
  if (!text) return;

  const bubble = document.createElement("div");
  bubble.className = "chat-message";
  bubble.dataset.sender = "self";
  bubble.innerHTML = `<span class="message-text"></span>`;
  bubble.querySelector(".message-text").textContent = text;

  document.getElementById("chat-log").appendChild(bubble);
  input.value = "";
});
