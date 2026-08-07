const form = document.querySelector("#summary-form");
const input = document.querySelector("#youtube-url");
const languageSelect = document.querySelector("#summary-language");
const button = document.querySelector("#submit-button");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const summaryEl = document.querySelector("#summary");
const keyPointsEl = document.querySelector("#key-points");
const keyPointsEmpty = document.querySelector("#key-points-empty");
const timestampsEl = document.querySelector("#timestamps");
const timestampsEmpty = document.querySelector("#timestamps-empty");
const metaEl = document.querySelector("#meta");
const videoPanel = document.querySelector("#video-panel");
const thumbnail = document.querySelector("#thumbnail");
const videoIdEl = document.querySelector("#video-id");
const fallbackBanner = document.querySelector("#fallback-banner");
const fallbackNotice = document.querySelector("#fallback-notice");
const studyNotesEl = document.querySelector("#study-notes");
const conceptsEl = document.querySelector("#important-concepts");
const extrasGrid = document.querySelector("#extras-grid");
const generationInfo = document.querySelector("#generation-info");
const genModelEl = document.querySelector("#gen-model");
const genTimeEl = document.querySelector("#gen-time");
const genLanguageEl = document.querySelector("#gen-language");

let currentVideoId = "";

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const url = input.value.trim();
  const language = languageSelect.value;
  const videoId = extractVideoId(url);

  if (videoId) {
    currentVideoId = videoId;
    showVideoPreview(videoId);
  }

  setLoading(true);
  setStatus("Reading transcript and preparing the summary...");
  resultsEl.hidden = true;
  fallbackBanner.hidden = true;

  try {
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, language }),
      signal: AbortSignal.timeout(120000)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not summarize this video.");
    }

    if (data.videoId) {
      currentVideoId = data.videoId;
    }

    renderResults(data);

    if (data.aiFallback && data.notice) {
      fallbackNotice.textContent = data.notice;
      fallbackBanner.hidden = false;
      setStatus("Summary ready (local fallback).");
    } else {
      setStatus("Summary ready.");
    }
  } catch (error) {
    if (error.name === "TimeoutError") {
      setStatus("Request timed out. The video might be too long or the server is busy. Try again.", true);
    } else {
      setStatus(error.message, true);
    }
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
  // Summary
  summaryEl.textContent = data.summary || "No summary returned.";

  // Meta line
  const parts = [];
  if (data.transcriptWordCount) parts.push(`${data.transcriptWordCount} words`);
  if (data.chunkCount) parts.push(`${data.chunkCount} chunk${data.chunkCount === 1 ? "" : "s"}`);
  metaEl.textContent = parts.join(" · ");

  // Key points
  const keyPoints = data.keyPoints || [];
  keyPointsEl.replaceChildren(
    ...keyPoints.map((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      return item;
    })
  );
  keyPointsEmpty.hidden = keyPoints.length > 0;

  // Timestamps (clickable YouTube links)
  const notes = data.timestampedNotes || [];
  const vid = currentVideoId || data.videoId || "";
  timestampsEl.replaceChildren(
    ...notes.map((entry) => {
      const item = document.createElement("li");

      if (vid) {
        const link = document.createElement("a");
        link.className = "time";
        link.href = `https://www.youtube.com/watch?v=${vid}&t=${Math.floor(entry.seconds)}`;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = entry.time;
        link.title = "Jump to this point in the video";
        item.append(link);
      } else {
        const time = document.createElement("span");
        time.className = "time";
        time.textContent = entry.time;
        item.append(time);
      }

      const note = document.createElement("span");
      note.className = "note";
      note.textContent = entry.note;
      item.append(note);

      return item;
    })
  );
  timestampsEmpty.hidden = notes.length > 0;

  // Study notes
  const studyNotes = data.studyNotes || [];
  if (studyNotes.length > 0) {
    studyNotesEl.replaceChildren(
      ...studyNotes.map((note) => {
        const item = document.createElement("li");
        item.textContent = note;
        return item;
      })
    );
  } else {
    studyNotesEl.replaceChildren();
  }

  // Important concepts
  const concepts = data.importantConcepts || [];
  if (concepts.length > 0) {
    conceptsEl.replaceChildren(
      ...concepts.map((concept) => {
        const item = document.createElement("li");
        item.textContent = concept;
        return item;
      })
    );
  } else {
    conceptsEl.replaceChildren();
  }

  // Show extras grid only if we have content
  extrasGrid.hidden = studyNotes.length === 0 && concepts.length === 0;

  // Generation info
  const genParts = [];
  if (data.model) genParts.push(`Model: ${data.model}`);
  if (data.generationTimeMs) genParts.push(`Time: ${(data.generationTimeMs / 1000).toFixed(1)}s`);
  if (data.languageLabel) genParts.push(`Language: ${data.languageLabel}`);

  if (genParts.length > 0) {
    genModelEl.textContent = data.model ? `🤖 ${data.model}` : "";
    genTimeEl.textContent = data.generationTimeMs ? `⏱ ${(data.generationTimeMs / 1000).toFixed(1)}s` : "";
    genLanguageEl.textContent = data.languageLabel ? `🌐 ${data.languageLabel}` : "";
    generationInfo.hidden = false;
  } else {
    generationInfo.hidden = true;
  }

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
  input.disabled = isLoading;
  languageSelect.disabled = isLoading;
  button.textContent = isLoading ? "Working..." : "Summarize";

  if (isLoading) {
    button.classList.add("loading");
  } else {
    button.classList.remove("loading");
  }
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
