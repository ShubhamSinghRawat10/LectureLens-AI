export function extractVideoId(input) {
  if (!input || typeof input !== "string") {
    throw new Error("Please paste a YouTube URL.");
  }

  const value = input.trim();

  if (/^[a-zA-Z0-9_-]{11}$/.test(value)) {
    return value;
  }

  try {
    const url = new URL(value);
    const host = url.hostname.replace(/^www\./, "");

    if (host === "youtu.be") {
      return cleanVideoId(url.pathname.slice(1));
    }

    if (host.endsWith("youtube.com")) {
      const fromQuery = url.searchParams.get("v");
      if (fromQuery) return cleanVideoId(fromQuery);

      const parts = url.pathname.split("/").filter(Boolean);
      const routeIndex = parts.findIndex((part) =>
        ["embed", "shorts", "live"].includes(part)
      );

      if (routeIndex >= 0 && parts[routeIndex + 1]) {
        return cleanVideoId(parts[routeIndex + 1]);
      }
    }
  } catch {
    const match = value.match(/(?:v=|youtu\.be\/|embed\/|shorts\/|live\/)([a-zA-Z0-9_-]{11})/);
    if (match?.[1]) return match[1];
  }

  throw new Error("That does not look like a supported YouTube URL.");
}

export function cleanTranscriptText(text = "") {
  return String(text)
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();
}

export function normalizeTranscript(rawTranscript) {
  // The youtube-transcript library returns { text, offset, duration, lang }.
  // InnerTube/srv3 path: offset & duration are in MILLISECONDS (large numbers).
  // Classic <text> path: offset & duration are in SECONDS (small numbers).
  // We detect which format by checking the max offset value.
  const isMilliseconds = detectMilliseconds(rawTranscript);

  return rawTranscript
    .map((item) => {
      const rawOffset = Number(item.offset) || 0;
      const rawDuration = Number(item.duration) || 0;
      const divisor = isMilliseconds ? 1000 : 1;

      return {
        text: cleanTranscriptText(item.text),
        start: Math.max(0, rawOffset / divisor),
        duration: Math.max(0, rawDuration / divisor)
      };
    })
    .filter((item) => item.text.length > 0);
}

function detectMilliseconds(rawTranscript) {
  // If any offset value is > 10000, it's almost certainly milliseconds.
  // A 10000-second offset would be ~2.7 hours, which is extremely rare,
  // while a 10000ms offset is just 10 seconds into the video.
  for (const item of rawTranscript) {
    const offset = Number(item.offset) || 0;
    if (offset > 10000) return true;
  }

  // Also check duration: ms durations are typically 2000-8000,
  // while second durations are typically 2-8.
  for (const item of rawTranscript) {
    const duration = Number(item.duration) || 0;
    if (duration > 500) return true;
  }

  return false;
}

export function transcriptToText(transcript) {
  return transcript.map((item) => item.text).join(" ");
}

export function formatTimestamp(seconds) {
  const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainingSeconds = safeSeconds % 60;

  if (hours > 0) {
    return [hours, minutes, remainingSeconds]
      .map((part) => String(part).padStart(2, "0"))
      .join(":");
  }

  return [minutes, remainingSeconds]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

export function chunkTranscript(transcript, maxWords = 2000) {
  const chunks = [];
  let currentSegments = [];
  let currentWordCount = 0;

  for (const segment of transcript) {
    const wordCount = segment.text.split(/\s+/).filter(Boolean).length;

    if (currentSegments.length > 0 && currentWordCount + wordCount > maxWords) {
      chunks.push(buildChunk(currentSegments));
      currentSegments = [];
      currentWordCount = 0;
    }

    currentSegments.push(segment);
    currentWordCount += wordCount;
  }

  if (currentSegments.length > 0) {
    chunks.push(buildChunk(currentSegments));
  }

  return chunks;
}

function buildChunk(segments) {
  return {
    start: segments[0]?.start ?? 0,
    end: (segments.at(-1)?.start ?? 0) + (segments.at(-1)?.duration ?? 0),
    text: segments
      .map((segment) => `[${formatTimestamp(segment.start)}] ${segment.text}`)
      .join("\n")
  };
}

function cleanVideoId(value) {
  const match = String(value).match(/[a-zA-Z0-9_-]{11}/);
  if (!match) {
    throw new Error("Could not find a valid YouTube video ID.");
  }

  return match[0];
}
