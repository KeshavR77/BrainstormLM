// BrainstormLM SSE client + thumbs/critique submitter (plan-06).

const SESSION_KEY = "brainstormlm.sessionId";

function getSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

const form = document.getElementById("ask-form");
const promptInput = document.getElementById("prompt");
const submitBtn = document.getElementById("submit-btn");
const useMemory = document.getElementById("use-memory");
const responseArea = document.getElementById("response-area");
const responseMeta = document.getElementById("response-meta");
const responseText = document.getElementById("response-text");
const feedbackControls = document.getElementById("feedback-controls");
const feedbackStatus = document.getElementById("feedback-status");
const critiqueInput = document.getElementById("critique");

let currentTrajectoryId = null;

async function streamAsk(prompt) {
  responseText.textContent = "";
  responseMeta.textContent = "";
  feedbackControls.hidden = true;
  feedbackStatus.textContent = "";
  responseArea.hidden = false;

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      session_id: getSessionId(),
      use_memory: useMemory.checked,
    }),
  });

  if (!res.ok) {
    const detail = await res.text();
    responseText.textContent = `Request failed (${res.status}): ${detail}`;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      handleSseBlock(block);
    }
  }
}

function handleSseBlock(block) {
  const lines = block.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  if (!data) return;
  let payload;
  try {
    payload = JSON.parse(data);
  } catch {
    return;
  }
  if (event === "meta") {
    currentTrajectoryId = payload.trajectory_id;
    responseMeta.textContent =
      `arm: ${payload.arm}` +
      (payload.adapter_path ? ` · adapter: ${payload.adapter_path}` : "");
  } else if (event === "token") {
    responseText.textContent += (responseText.textContent ? " " : "") + payload.text;
  } else if (event === "done") {
    feedbackControls.hidden = false;
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;
  submitBtn.disabled = true;
  try {
    await streamAsk(text);
  } finally {
    submitBtn.disabled = false;
  }
});

document.querySelectorAll(".feedback-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!currentTrajectoryId) return;
    const thumbs = btn.dataset.thumbs;
    const critique = critiqueInput.value.trim() || null;
    feedbackStatus.textContent = "submitting…";
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trajectory_id: currentTrajectoryId,
        thumbs,
        critique,
      }),
    });
    feedbackStatus.textContent = res.ok ? "thanks!" : "submit failed";
  });
});
