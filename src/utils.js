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
  return rawTranscript
    .map((item) => {
      const hasStartSeconds = Number.isFinite(item.start);
      const start = hasStartSeconds ? item.start : (Number(item.offset) || 0) / 1000;
      const duration = hasStartSeconds
        ? Number(item.duration) || 0
        : (Number(item.duration) || 0) / 1000;

      return {
        text: cleanTranscriptText(item.text),
        start: Math.max(0, start),
        duration: Math.max(0, duration)
      };
    })
    .filter((item) => item.text.length > 0);
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

export function chunkTranscript(transcript, maxWords = 900) {
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
