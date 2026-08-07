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

// ---------------------------------------------------------------------------
// Startup validation
// ---------------------------------------------------------------------------
function validateStartup() {
  const provider = (process.env.AI_PROVIDER || "").toLowerCase() || "auto";
  const geminiKey = process.env.GEMINI_API_KEY || "";
  const groqKey = process.env.GROQ_API_KEY || "";
  const geminiModel = process.env.GEMINI_MODEL || "gemini-2.5-flash";
  const fallbackModel = process.env.GEMINI_FALLBACK_MODEL || "gemini-2.0-flash";
  const groqModel = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
  const localFallback = process.env.LOCAL_SUMMARY_FALLBACK !== "false";

  console.log("\n╔══════════════════════════════════════╗");
  console.log("║     LectureLens AI — Startup Check   ║");
  console.log("╚══════════════════════════════════════╝\n");

  const check = (ok, label) => console.log(`  ${ok ? "✓" : "✗"} ${label}`);

  check(true, `Provider       = ${provider}`);
  check(geminiKey.length > 0, `Gemini Key     = ${geminiKey ? "loaded (" + geminiKey.slice(0, 8) + "...)" : "MISSING"}`);
  check(groqKey.length > 0 || provider !== "groq", `Groq Key       = ${groqKey ? "loaded" : provider === "groq" ? "MISSING" : "not needed"}`);
  check(true, `Gemini Model   = ${geminiModel}`);
  check(true, `Fallback Model = ${fallbackModel}`);
  check(true, `Groq Model     = ${groqModel}`);
  check(true, `Local Fallback = ${localFallback ? "enabled" : "disabled"}`);
  check(true, `Port           = ${port}`);

  console.log("");

  // Critical warnings
  if (provider === "gemini" && !geminiKey) {
    console.error("  ⚠  GEMINI_API_KEY is missing. Gemini requests will fail.\n");
  }
  if (provider === "groq" && !groqKey) {
    console.error("  ⚠  GROQ_API_KEY is missing. Groq requests will fail.\n");
  }
}

// ---------------------------------------------------------------------------
// Rate limiting (simple in-memory, per IP)
// ---------------------------------------------------------------------------
const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 10;

function rateLimit(request, response, next) {
  const ip = request.ip || request.socket.remoteAddress || "unknown";
  const now = Date.now();
  const entry = rateLimitMap.get(ip);

  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW_MS) {
    rateLimitMap.set(ip, { windowStart: now, count: 1 });
    return next();
  }

  entry.count += 1;

  if (entry.count > RATE_LIMIT_MAX) {
    return response.status(429).json({
      error: "Too many requests. Please wait a minute before trying again."
    });
  }

  next();
}

// Clean up stale rate limit entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimitMap) {
    if (now - entry.windowStart > RATE_LIMIT_WINDOW_MS * 2) {
      rateLimitMap.delete(ip);
    }
  }
}, 300_000).unref();

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.static(publicDir));

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------
app.get("/api/health", (_request, response) => {
  response.json({ ok: true });
});

app.post("/api/summarize", rateLimit, async (request, response) => {
  const startTime = Date.now();

  try {
    // Input validation
    const { url, language } = request.body || {};

    if (!url || typeof url !== "string" || url.trim().length === 0) {
      return response.status(400).json({
        error: "Please provide a YouTube URL."
      });
    }

    if (url.trim().length > 2048) {
      return response.status(400).json({
        error: "URL is too long."
      });
    }

    const videoId = extractVideoId(url);
    const rawTranscript = await fetchTranscript(videoId);
    const transcript = normalizeTranscript(rawTranscript);

    if (transcript.length === 0) {
      return response.status(404).json({
        error: "No transcript text was found for this video."
      });
    }

    const result = await summarizeTranscript(transcript, videoId, language);
    const generationTimeMs = Date.now() - startTime;

    response.json({
      videoId,
      transcriptPreview: transcript.slice(0, 5),
      transcriptWordCount: transcriptToText(transcript).split(/\s+/).filter(Boolean).length,
      generationTimeMs,
      ...result
    });
  } catch (error) {
    console.error("[API] /api/summarize error:", error instanceof Error ? error.message : error);
    const status = statusFromError(error);
    response.status(status).json({
      error: readableError(error),
      generationTimeMs: Date.now() - startTime
    });
  }
});

app.get("/{*path}", (_request, response) => {
  response.sendFile(path.join(publicDir, "index.html"));
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
validateStartup();

app.listen(port, () => {
  console.log(`LectureLens AI is running at http://localhost:${port}\n`);
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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
