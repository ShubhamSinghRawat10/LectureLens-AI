const form = document.querySelector("#summary-form");
const input = document.querySelector("#youtube-url");
const languageSelect = document.querySelector("#summary-language");
const button = document.querySelector("#submit-button");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const summaryEl = document.querySelector("#summary");
const keyPointsEl = document.querySelector("#key-points");
const timestampsEl = document.querySelector("#timestamps");
const metaEl = document.querySelector("#meta");
const videoPanel = document.querySelector("#video-panel");
const thumbnail = document.querySelector("#thumbnail");
const videoIdEl = document.querySelector("#video-id");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const url = input.value.trim();
  const language = languageSelect.value;
  const videoId = extractVideoId(url);

  if (videoId) {
    showVideoPreview(videoId);
  }

  setLoading(true);
  setStatus("Reading transcript and preparing the summary...");
  resultsEl.hidden = true;

  try {
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, language })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not summarize this video.");
    }

    renderResults(data);
    setStatus(data.notice ? `Summary ready. ${data.notice}` : "Summary ready.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setLoading(false);
  }
});

input.addEventListener("input", () => {
  const videoId = extractVideoId(input.value);
  if (videoId) {
    showVideoPreview(videoId);
  }
});

function renderResults(data) {
  summaryEl.textContent = data.summary || "No summary returned.";
  metaEl.textContent = `${data.transcriptWordCount || 0} words - ${data.chunkCount || 1} chunk${
    data.chunkCount === 1 ? "" : "s"
  } - ${data.languageLabel || "Auto"} - ${data.model || "model unknown"}`;

  keyPointsEl.replaceChildren(
    ...(data.keyPoints || []).map((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      return item;
    })
  );

  timestampsEl.replaceChildren(
    ...(data.timestampedNotes || []).map((entry) => {
      const item = document.createElement("li");
      const time = document.createElement("span");
      const note = document.createElement("span");

      time.className = "time";
      note.className = "note";
      time.textContent = entry.time;
      note.textContent = entry.note;

      item.append(time, note);
      return item;
    })
  );

  if (data.videoId) {
    showVideoPreview(data.videoId);
  }

  resultsEl.hidden = false;
}

function showVideoPreview(videoId) {
  thumbnail.src = `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`;
  videoIdEl.textContent = videoId;
  videoPanel.hidden = false;
}

function setLoading(isLoading) {
  button.disabled = isLoading;
  button.textContent = isLoading ? "Working..." : "Summarize";
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function extractVideoId(value) {
  const direct = value.trim().match(/^[a-zA-Z0-9_-]{11}$/);
  if (direct) return direct[0];

  try {
    const url = new URL(value);
    const host = url.hostname.replace(/^www\./, "");

    if (host === "youtu.be") {
      return url.pathname.slice(1).match(/[a-zA-Z0-9_-]{11}/)?.[0] || "";
    }

    if (host.endsWith("youtube.com")) {
      const fromQuery = url.searchParams.get("v");
      if (fromQuery) return fromQuery.match(/[a-zA-Z0-9_-]{11}/)?.[0] || "";

      return url.pathname.match(/\/(?:embed|shorts|live)\/([a-zA-Z0-9_-]{11})/)?.[1] || "";
    }
  } catch {
    return value.match(/(?:v=|youtu\.be\/|embed\/|shorts\/|live\/)([a-zA-Z0-9_-]{11})/)?.[1] || "";
  }

  return "";
}
