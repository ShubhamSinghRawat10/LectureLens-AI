import "dotenv/config";

import cors from "cors";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fetchTranscript } from "youtube-transcript";

import { extractVideoId, normalizeTranscript, transcriptToText } from "./utils.js";
import { summarizeTranscript } from "./summarizer.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicDir = path.join(__dirname, "..", "public");

const app = express();
const port = Number(process.env.PORT) || 3000;

app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.static(publicDir));

app.get("/api/health", (_request, response) => {
  response.json({ ok: true });
});

app.post("/api/summarize", async (request, response) => {
  try {
    const { url, language } = request.body;
    const videoId = extractVideoId(url);
    const rawTranscript = await fetchTranscript(videoId);
    const transcript = normalizeTranscript(rawTranscript);

    if (transcript.length === 0) {
      return response.status(404).json({
        error: "No transcript text was found for this video."
      });
    }

    const result = await summarizeTranscript(transcript, videoId, language);

    response.json({
      videoId,
      transcriptPreview: transcript.slice(0, 5),
      transcriptWordCount: transcriptToText(transcript).split(/\s+/).filter(Boolean).length,
      ...result
    });
  } catch (error) {
    const status = statusFromError(error);
    response.status(status).json({
      error: readableError(error)
    });
  }
});

app.get(/.*/, (_request, response) => {
  response.sendFile(path.join(publicDir, "index.html"));
});

app.listen(port, () => {
  console.log(`LectureLens AI is running at http://localhost:${port}`);
});

function statusFromError(error) {
  const message = readableError(error).toLowerCase();

  if (message.includes("api key")) return 500;
  if (message.includes("youtube") || message.includes("transcript")) return 404;
  if (message.includes("url")) return 400;

  return 500;
}

function readableError(error) {
  if (error instanceof Error && error.message) {
    const parsedGeminiError = parseGeminiError(error.message);
    if (parsedGeminiError) return parsedGeminiError;

    if (error.message.toLowerCase().includes("transcript is disabled")) {
      return "Transcript is disabled for this YouTube video. Try a video that has captions or an available transcript.";
    }

    return error.message;
  }

  return "Something went wrong while summarizing this lecture.";
}

function parseGeminiError(message) {
  const jsonStart = message.indexOf("{");
  if (jsonStart === -1) return "";

  try {
    const payload = JSON.parse(message.slice(jsonStart));
    const status = payload?.error?.status;
    const code = payload?.error?.code;
    const text = payload?.error?.message || "";

    if (code === 503 || status === "UNAVAILABLE") {
      return "Gemini is temporarily overloaded. Please wait a minute and try again.";
    }

    if (code === 429 || status === "RESOURCE_EXHAUSTED") {
      return "Gemini API quota is exhausted for this key. Check your Google AI Studio limits or try again later.";
    }

    return text;
  } catch {
    return "";
  }
}
