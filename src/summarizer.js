import { GoogleGenAI, Type } from "@google/genai";
import { chunkTranscript, formatTimestamp } from "./utils.js";

const DEFAULT_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || "gemini-2.0-flash";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions";

let client;
let lastUsedModel = configuredProvider() === "groq" ? GROQ_MODEL : DEFAULT_MODEL;

function getClient() {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is missing. Add it to a .env file first.");
  }

  if (!client) {
    client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }

  return client;
}

export async function summarizeTranscript(transcript, videoId, language = "english") {
  const chunks = chunkTranscript(transcript);
  const targetLanguage = normalizeSummaryLanguage(language);

  if (chunks.length === 0) {
    throw new Error("Transcript is empty.");
  }

  if (configuredProvider() === "local") {
    return await createLocalSummary(
      transcript,
      chunks.length,
      new Error("Local summary provider selected."),
      targetLanguage
    );
  }

  try {
    const partials = [];

    for (let index = 0; index < chunks.length; index += 1) {
      const partial = await summarizeChunk(chunks[index], index + 1, chunks.length, targetLanguage);
      partials.push(partial);
    }

    const result =
      partials.length === 1
        ? partials[0]
        : await combineChunkSummaries(partials, videoId, targetLanguage);

    return normalizeSummary(result, chunks.length, targetLanguage);
  } catch (error) {
    if (process.env.LOCAL_SUMMARY_FALLBACK === "false") {
      throw error;
    }

    return await createLocalSummary(transcript, chunks.length, error, targetLanguage);
  }
}

async function summarizeChunk(chunk, chunkNumber, totalChunks, targetLanguage) {
  const content = await requestJsonSummary([
    {
      role: "developer",
      content:
        `You are an expert lecture summarizer. Your job is to create highly accurate, faithful summaries from video transcripts.

CRITICAL RULES:
- ONLY use information that is EXPLICITLY stated in the transcript. Never invent, guess, or add information not present.
- Every key point MUST come directly from something said in the transcript.
- Timestamps must match the ones shown in the transcript. NEVER invent timestamps.
- Be specific and detailed — avoid vague generic statements.
- Write every user-facing value in ${targetLanguage.promptName}.
- Return ONLY valid JSON, no markdown, no extra text.`
    },
    {
      role: "user",
      content: `Carefully read this transcript chunk (${chunkNumber} of ${totalChunks}) and create an accurate JSON summary.

The JSON object MUST have:
- "summary": A clear, detailed paragraph that accurately captures what is taught/discussed in this chunk. Be specific about the actual topics and concepts mentioned.
- "keyPoints": An array of 5 to 8 concise but specific strings. Each key point must directly correspond to something explicitly stated or explained in the transcript. Do NOT write generic/obvious statements.
- "timestampedNotes": An array of objects with "time" (string like "MM:SS"), "seconds" (number), and "note" (string). Pick the most important moments. Use ONLY timestamps that appear in the transcript.

Output language: ${targetLanguage.promptName}.

Transcript:
${chunk.text}`
    }
  ]);

  return content;
}

async function combineChunkSummaries(partials, videoId, targetLanguage) {
  const content = await requestJsonSummary([
    {
      role: "developer",
      content:
        `You are an expert lecture summarizer that combines partial summaries into one cohesive, accurate result.

CRITICAL RULES:
- ONLY include information that appears in the chunk summaries. Never add new information.
- Deduplicate key points but keep them specific and detailed.
- Preserve all timestamps exactly as they appear. NEVER invent new timestamps.
- The final summary must faithfully represent the actual content of the lecture.
- Write every user-facing value in ${targetLanguage.promptName}.
- Return ONLY valid JSON, no markdown, no extra text.`
    },
    {
      role: "user",
      content: `Combine these ${partials.length} chunk summaries for YouTube video ${videoId} into one comprehensive result.

The JSON object MUST have:
- "summary": A cohesive 2 to 4 paragraph summary that flows naturally and covers ALL the main topics discussed in the lecture. Be specific about actual concepts taught.
- "keyPoints": An array of 7 to 12 deduplicated, specific key points. Each must reflect actual content from the lecture. Remove redundancy but keep detail.
- "timestampedNotes": An array of the most important timestamped objects with "time", "seconds", and "note". Keep timestamps exactly from the chunk summaries.

Output language: ${targetLanguage.promptName}.

Chunk summaries:
${JSON.stringify(partials, null, 2)}`
    }
  ]);

  return content;
}

async function requestJsonSummary(messages) {
  if (configuredProvider() === "groq") {
    return requestGroqJsonSummary(messages);
  }

  const systemInstruction = messages
    .filter((message) => message.role === "developer")
    .map((message) => message.content)
    .join("\n");
  const contents = messages
    .filter((message) => message.role !== "developer")
    .map((message) => message.content)
    .join("\n\n");

  const response = await generateWithFallback({
    contents,
    systemInstruction
  });

  const raw = response.text;
  if (!raw) {
    throw new Error("Gemini returned an empty response.");
  }

  return parseJson(raw);
}

async function requestGroqJsonSummary(messages) {
  if (!process.env.GROQ_API_KEY) {
    throw new Error("GROQ_API_KEY is missing. Add it to a .env file first.");
  }

  let lastError;

  for (let attempt = 0; attempt < 3; attempt++) {
    const response = await fetch(GROQ_BASE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: messages.map((message) => ({
          role: message.role === "developer" ? "system" : message.role,
          content: message.content
        })),
        temperature: 0.1,
        max_tokens: 4096,
        top_p: 0.9,
        response_format: { type: "json_object" }
      })
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      lastError = new Error(JSON.stringify(payload));
      console.error(`[Groq] Attempt ${attempt + 1} failed: ${response.status}`);

      if (response.status === 429 && attempt < 2) {
        const delay = Math.min(15000 * Math.pow(2, attempt), 60000);
        console.log(`[Groq] Rate limited. Retrying in ${delay / 1000}s...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }

      throw lastError;
    }

    const raw = payload?.choices?.[0]?.message?.content;
    if (!raw) {
      throw new Error("Groq returned an empty response.");
    }

    lastUsedModel = `groq/${GROQ_MODEL}`;
    return parseJson(raw);
  }

  throw lastError;
}

async function generateWithFallback({ contents, systemInstruction }) {
  const models = [...new Set([DEFAULT_MODEL, FALLBACK_MODEL].filter(Boolean))];
  let lastError;

  for (const model of models) {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const response = await getClient().models.generateContent({
          model,
          contents,
          config: {
            systemInstruction,
            responseMimeType: "application/json",
            responseSchema: summarySchema()
          }
        });
        lastUsedModel = model;
        return response;
      } catch (error) {
        lastError = error;
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error(`[Gemini] ${model} attempt ${attempt + 1} failed: ${errorMsg.slice(0, 200)}`);

        if (isRateLimitError(error) && attempt < 2) {
          const delay = Math.min(15000 * Math.pow(2, attempt), 60000);
          console.log(`[Gemini] Rate limited. Retrying in ${delay / 1000}s...`);
          await sleep(delay);
          continue;
        }

        if (!shouldTryFallbackModel(error)) {
          throw error;
        }

        break;
      }
    }
  }

  throw lastError;
}

function isRateLimitError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes('"code":429') ||
    message.includes('RESOURCE_EXHAUSTED') ||
    message.toLowerCase().includes('quota exceeded') ||
    message.toLowerCase().includes('rate limit')
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function shouldTryFallbackModel(error) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("\"code\":503") ||
    message.includes("\"code\":429") ||
    message.includes("UNAVAILABLE") ||
    message.includes("RESOURCE_EXHAUSTED") ||
    message.toLowerCase().includes("high demand") ||
    message.toLowerCase().includes("quota exceeded")
  );
}

function configuredProvider() {
  if (process.env.AI_PROVIDER) {
    return process.env.AI_PROVIDER.toLowerCase();
  }

  if (process.env.GROQ_API_KEY) return "groq";
  if (process.env.GEMINI_API_KEY) return "gemini";
  return "local";
}

function summarySchema() {
  return {
    type: Type.OBJECT,
    properties: {
      summary: { type: Type.STRING },
      keyPoints: {
        type: Type.ARRAY,
        items: { type: Type.STRING }
      },
      timestampedNotes: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            time: { type: Type.STRING },
            seconds: { type: Type.NUMBER },
            note: { type: Type.STRING }
          },
          required: ["time", "seconds", "note"]
        }
      }
    },
    required: ["summary", "keyPoints", "timestampedNotes"]
  };
}

function parseJson(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    const match = raw.match(/\{[\s\S]*\}/);
    if (!match) {
      throw new Error("Gemini response was not valid JSON.");
    }

    return JSON.parse(match[0]);
  }
}

function normalizeSummary(result, chunkCount, targetLanguage) {
  const timestampedNotes = Array.isArray(result.timestampedNotes)
    ? result.timestampedNotes
        .map((item) => {
          const seconds = Number(item.seconds) || parseTimestamp(item.time);
          return {
            time: item.time || formatTimestamp(seconds),
            seconds,
            note: String(item.note || "").trim()
          };
        })
        .filter((item) => item.note)
        .sort((a, b) => a.seconds - b.seconds)
    : [];

  return {
    summary: String(result.summary || "").trim(),
    keyPoints: Array.isArray(result.keyPoints)
      ? result.keyPoints.map((point) => String(point).trim()).filter(Boolean)
      : [],
    timestampedNotes,
    chunkCount,
    language: targetLanguage.id,
    languageLabel: targetLanguage.label,
    model: lastUsedModel
  };
}

function parseTimestamp(value = "") {
  const parts = String(value)
    .split(":")
    .map((part) => Number(part));

  if (parts.some((part) => Number.isNaN(part))) return 0;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parts[0] || 0;
}

function normalizeSummaryLanguage(language) {
  const id = String(language || "english").toLowerCase().trim();
  return SUMMARY_LANGUAGES[id] || SUMMARY_LANGUAGES.english;
}

function localSummaryLead(targetLanguage, focus) {
  const topics = joinHumanList(focus);

  switch (targetLanguage.id) {
    case "hindi":
      return `Is lecture ka main focus ${topics} par hai.`;
    default:
      return `This lecture mainly focuses on ${topics}.`;
  }
}

function localGenericLead(targetLanguage) {
  switch (targetLanguage.id) {
    case "hindi":
      return "Yeh lecture available YouTube transcript se summarize kiya gaya hai.";
    default:
      return "This lecture was summarized from the available YouTube transcript.";
  }
}

function localFallbackNotice(targetLanguage, usedTranslator) {
  if (targetLanguage.id === "english" && !usedTranslator) {
    return "AI model was unavailable, so this was generated locally in English from the transcript.";
  }

  if (usedTranslator) {
    return `AI model was unavailable, so this was generated locally and translated to ${targetLanguage.label}.`;
  }

  return `AI model was unavailable, so this was generated locally from the transcript. ${targetLanguage.label} translation could not run.`;
}

async function createLocalSummary(
  transcript,
  chunkCount,
  error,
  targetLanguage = normalizeSummaryLanguage("english")
) {
  const windows = buildTranscriptWindows(transcript);
  const conceptSummary = await createConceptLocalSummary(
    windows,
    chunkCount,
    error,
    targetLanguage
  );

  if (conceptSummary) {
    return conceptSummary;
  }

  const keywords = topKeywords(transcript.map((item) => item.text).join(" "));
  const ranked = windows
    .map((window) => ({
      ...window,
      score: scoreText(window.text, keywords)
    }))
    .sort((a, b) => b.score - a.score);

  const importantWindows = ranked.slice(0, 8).sort((a, b) => a.seconds - b.seconds);
  const summaryWindows = ranked.slice(0, 3).sort((a, b) => a.seconds - b.seconds);
  const focus = keywords.slice(0, 5).map((item) => item.word);
  const summaryLead =
    focus.length > 0
      ? localSummaryLead(targetLanguage, focus)
      : localGenericLead(targetLanguage);
  const summaryDetails = summaryWindows
    .map((window) => limitText(window.text, 240))
    .join(" ");
  const translated = await translateLocalResult(
    {
      summary: `${summaryLead}\n\n${summaryDetails}`,
      keyPoints: importantWindows.slice(0, 7).map((window) => limitText(window.text, 170)),
      timestampedNotes: importantWindows.map((window) => ({
        time: formatTimestamp(window.seconds),
        seconds: window.seconds,
        note: limitText(window.text, 180)
      }))
    },
    targetLanguage
  );

  return {
    summary: translated.summary,
    keyPoints: translated.keyPoints,
    timestampedNotes: translated.timestampedNotes,
    chunkCount,
    language: targetLanguage.id,
    languageLabel: targetLanguage.label,
    model: "local-transcript-fallback",
    aiFallback: true,
    notice: `${localFallbackNotice(targetLanguage, translated.usedTranslator)} ${summarizeProviderError(error)}`
  };
}

async function createConceptLocalSummary(windows, chunkCount, error, targetLanguage) {
  const analysisWindows = await buildAnalysisWindows(windows);
  const detectedTopics = detectLectureTopics(analysisWindows);

  if (detectedTopics.length < 2) {
    return null;
  }

  const topicLabels = detectedTopics.map((topic) => topic.label);
  const summary = [
    `This lecture is a broad CS fundamentals revision covering ${joinHumanList(topicLabels)}.`,
    detectedTopics
      .map((topic) => `${topic.label}: ${topic.summary}`)
      .join("\n")
  ].join("\n\n");
  const keyPoints = [
    ...detectedTopics.flatMap((topic) => topic.keyPoints.slice(0, 2)),
    ...detectedTopics.flatMap((topic) => topic.keyPoints.slice(2))
  ].slice(0, 10);
  const timestampedNotes = detectedTopics.flatMap((topic) =>
    topic.notes.map((note) => ({
      time: formatTimestamp(note.seconds),
      seconds: note.seconds,
      note: `${topic.label}: ${note.text}`
    }))
  );
  const translated = await translateLocalResult(
    {
      summary,
      keyPoints,
      timestampedNotes
    },
    targetLanguage,
    { alreadyEnglish: true }
  );

  return {
    summary: translated.summary,
    keyPoints: translated.keyPoints,
    timestampedNotes: translated.timestampedNotes,
    chunkCount,
    language: targetLanguage.id,
    languageLabel: targetLanguage.label,
    model: "local-concept-fallback",
    aiFallback: true,
    notice: `${localFallbackNotice(targetLanguage, translated.usedTranslator)} ${summarizeProviderError(error)}`
  };
}

async function buildAnalysisWindows(windows) {
  const candidates = selectRepresentativeWindows(windows, 32);
  const translated = [];

  for (const window of candidates) {
    const rawText = limitText(window.text, 1200);
    const englishText = await translateText(rawText, "en").catch(() => rawText);
    const text = cleanLocalText(englishText);

    if (!isNoiseText(text)) {
      translated.push({
        seconds: window.seconds,
        text
      });
    }
  }

  return translated;
}

function selectRepresentativeWindows(windows, maxCount) {
  if (windows.length <= maxCount) return windows;

  const buckets = Math.min(maxCount, Math.max(8, Math.ceil(windows.length / 6)));
  const selected = [];

  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const start = Math.floor((bucket * windows.length) / buckets);
    const end = Math.max(start + 1, Math.floor(((bucket + 1) * windows.length) / buckets));
    const slice = windows.slice(start, end);
    const best = slice
      .map((window) => ({
        ...window,
        score: localEducationScore(window.text) - localNoiseScore(window.text)
      }))
      .sort((a, b) => b.score - a.score)[0];

    if (best) selected.push(best);
  }

  return selected;
}

function detectLectureTopics(analysisWindows) {
  return TOPIC_DEFINITIONS.map((definition) => {
    const hits = analysisWindows
      .map((window) => ({
        ...window,
        score: topicScore(window.text, definition)
      }))
      .filter((window) => window.score > 0)
      .sort((a, b) => b.score - a.score);

    const firstHit = hits.slice().sort((a, b) => a.seconds - b.seconds)[0];
    const bestHits = hits
      .slice(0, 3)
      .sort((a, b) => a.seconds - b.seconds)
      .map((window) => ({
        seconds: window.seconds,
        text: bestEvidenceSentence(window.text, definition)
      }));

    return {
      ...definition,
      score: hits.reduce((sum, hit) => sum + hit.score, 0),
      seconds: firstHit?.seconds ?? Number.MAX_SAFE_INTEGER,
      notes: bestHits.length > 0 ? bestHits : [{ seconds: firstHit?.seconds ?? 0, text: definition.summary }]
    };
  })
    .filter((topic) => topic.score >= topic.threshold)
    .sort((a, b) => a.seconds - b.seconds);
}

function topicScore(text, definition) {
  const lower = text.toLowerCase();
  return definition.terms.reduce((score, term) => {
    return lower.includes(term) ? score + 1 : score;
  }, 0);
}

function bestEvidenceSentence(text, definition) {
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const best = sentences
    .map((sentence) => ({
      sentence,
      score: topicScore(sentence, definition)
    }))
    .sort((a, b) => b.score - a.score)[0]?.sentence;

  return limitText(best || text, 170);
}

function localEducationScore(text) {
  const lower = text.toLowerCase();
  const terms = [
    "operating",
    "process",
    "thread",
    "cpu",
    "memory",
    "deadlock",
    "scheduling",
    "server",
    "client",
    "network",
    "http",
    "ip",
    "mac",
    "class",
    "object",
    "function",
    "database",
    "table",
    "query",
    "sql",
    "example",
    "means",
    "what is"
  ];

  return terms.reduce((score, term) => (lower.includes(term) ? score + 2 : score), 0);
}

function localNoiseScore(text) {
  const lower = text.toLowerCase();
  const terms = [
    "telegram",
    "whatsapp",
    "channel",
    "subscribe",
    "description",
    "comment",
    "notes",
    "doubt",
    "homework",
    "like this video"
  ];

  return terms.reduce((score, term) => (lower.includes(term) ? score + 5 : score), 0);
}

function isNoiseText(text) {
  return localNoiseScore(text) > localEducationScore(text) + 4;
}

async function translateLocalResult(result, targetLanguage, options = {}) {
  const targetCode = targetLanguage.translateCode;
  if (!targetCode || (options.alreadyEnglish && targetLanguage.id === "english")) {
    return { ...result, usedTranslator: false };
  }

  try {
    const summary = await translateText(result.summary, targetCode);
    const keyPoints = [];
    for (const point of result.keyPoints) {
      keyPoints.push(await translateText(point, targetCode));
    }

    const timestampedNotes = [];
    for (const note of result.timestampedNotes) {
      timestampedNotes.push({
        ...note,
        note: await translateText(note.note, targetCode)
      });
    }

    return {
      summary,
      keyPoints,
      timestampedNotes,
      usedTranslator: true
    };
  } catch {
    return { ...result, usedTranslator: false };
  }
}

async function translateText(text, targetCode) {
  const clean = String(text || "").trim();
  if (!clean) return "";

  const url = new URL("https://translate.googleapis.com/translate_a/single");
  url.searchParams.set("client", "gtx");
  url.searchParams.set("sl", "auto");
  url.searchParams.set("tl", targetCode);
  url.searchParams.set("dt", "t");
  url.searchParams.set("q", clean);

  const response = await fetch(url, {
    signal: AbortSignal.timeout(10000)
  });

  if (!response.ok) {
    throw new Error("Translation request failed.");
  }

  const payload = await response.json();
  const translated = payload?.[0]?.map((part) => part?.[0] || "").join("").trim();

  return translated || clean;
}

function buildTranscriptWindows(transcript) {
  const windows = [];
  let current = [];
  let wordCount = 0;

  for (const item of transcript) {
    const text = cleanLocalText(item.text);
    if (!text) continue;

    const words = text.split(/\s+/).filter(Boolean).length;
    const shouldFlush = current.length > 0 && (wordCount + words > 85 || current.length >= 8);

    if (shouldFlush) {
      windows.push(buildLocalWindow(current));
      current = [];
      wordCount = 0;
    }

    current.push({ ...item, text });
    wordCount += words;
  }

  if (current.length > 0) {
    windows.push(buildLocalWindow(current));
  }

  return windows;
}

function buildLocalWindow(items) {
  return {
    seconds: items[0]?.start ?? 0,
    text: items.map((item) => item.text).join(" ")
  };
}

function topKeywords(text) {
  const counts = new Map();

  for (const token of text.toLowerCase().match(/[a-z0-9][a-z0-9-']+/g) || []) {
    if (token.length < 4 || STOP_WORDS.has(token)) continue;
    counts.set(token, (counts.get(token) || 0) + 1);
  }

  return [...counts.entries()]
    .map(([word, count]) => ({ word, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20);
}

function scoreText(text, keywords) {
  const lower = text.toLowerCase();
  const keywordScore = keywords.reduce((score, keyword, index) => {
    return lower.includes(keyword.word) ? score + Math.max(1, 20 - index) : score;
  }, 0);
  const lengthScore = Math.min(25, text.split(/\s+/).length / 3);

  return keywordScore + lengthScore;
}

function cleanLocalText(text) {
  return String(text)
    .replace(/\[[^\]]+\]/g, " ")
    .replace(/\([^)]*music[^)]*\)/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function limitText(text, maxLength) {
  const clean = cleanLocalText(text);
  if (clean.length <= maxLength) return ensureSentence(clean);

  const trimmed = clean.slice(0, maxLength).replace(/\s+\S*$/, "");
  return ensureSentence(`${trimmed}...`);
}

function ensureSentence(text) {
  if (!text) return "";
  const capitalized = text[0].toUpperCase() + text.slice(1);
  return /[.!?]$/.test(capitalized) ? capitalized : `${capitalized}.`;
}

function joinHumanList(items) {
  if (items.length <= 1) return items.join("");
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items.at(-1)}`;
}

function summarizeProviderError(error) {
  const message = error instanceof Error ? error.message : String(error || "");
  const lower = message.toLowerCase();

  if (lower.includes("quota") || lower.includes("429")) {
    return "Reason: provider quota/rate limit.";
  }

  if (lower.includes("high demand") || lower.includes("503") || lower.includes("unavailable")) {
    return "Reason: provider overloaded.";
  }

  if (lower.includes("api key")) {
    return "Reason: API key issue.";
  }

  return "Reason: provider request failed.";
}

const STOP_WORDS = new Set([
  "about",
  "above",
  "after",
  "again",
  "also",
  "because",
  "been",
  "before",
  "being",
  "between",
  "could",
  "does",
  "doing",
  "down",
  "during",
  "each",
  "from",
  "have",
  "here",
  "into",
  "just",
  "like",
  "more",
  "most",
  "much",
  "need",
  "only",
  "other",
  "over",
  "same",
  "should",
  "some",
  "such",
  "than",
  "that",
  "their",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "through",
  "very",
  "want",
  "what",
  "when",
  "where",
  "which",
  "while",
  "with",
  "will",
  "would",
  "your"
]);

const TOPIC_DEFINITIONS = [
  {
    label: "Operating Systems",
    threshold: 3,
    terms: [
      "operating system",
      "os",
      "kernel",
      "resource allocator",
      "control program",
      "process",
      "thread",
      "cpu",
      "scheduling",
      "memory",
      "paging",
      "deadlock",
      "semaphore",
      "mutex",
      "page fault"
    ],
    summary:
      "the teacher explains what an OS does, why it is needed between user applications and hardware, types of kernels/operating systems, process and thread management, CPU scheduling, synchronization, deadlocks, semaphores/mutexes, and memory management topics such as paging and page faults.",
    keyPoints: [
      "An operating system is explained as a resource allocator and control program that manages CPU, memory, hardware, files, and user programs.",
      "The lecture covers process management: program vs process, process states, threads, user-level vs kernel-level threads, and why process isolation matters.",
      "CPU scheduling and synchronization are discussed through concepts like critical section, semaphores, mutexes, and deadlock handling.",
      "Memory management is revised with paging, page faults, virtual memory ideas, and the overhead of managing many processes."
    ]
  },
  {
    label: "Computer Networks",
    threshold: 2,
    terms: [
      "network",
      "server",
      "client",
      "internet",
      "http",
      "protocol",
      "ssl",
      "ip",
      "subnet",
      "subnetting",
      "gateway",
      "modem",
      "nic",
      "mac address",
      "wi-fi",
      "wifi"
    ],
    summary:
      "the teacher moves into networking with a practical client-server explanation using YouTube/web requests, HTTP and security layers, IP addressing, subnetting, gateways, modems, NIC cards, MAC addresses, and how data reaches a device.",
    keyPoints: [
      "Networking is explained through real examples: a phone or browser requests data from a server and receives it through network devices and protocols.",
      "The lecture covers IP addressing and subnetting to divide large networks into smaller manageable parts.",
      "Important network devices and layers are revised, including gateway, modem, NIC, MAC address, Wi-Fi, HTTP, and security/SSL layers."
    ]
  },
  {
    label: "Object-Oriented Programming",
    threshold: 2,
    terms: [
      "object",
      "class",
      "property",
      "behavior",
      "function",
      "method",
      "constructor",
      "this",
      "inheritance",
      "encapsulation",
      "abstraction",
      "polymorphism",
      "overloading",
      "overriding"
    ],
    summary:
      "the OOP section explains real-world objects, classes as blueprints, properties and behavior, methods/functions, constructors, the this keyword, and interview concepts such as inheritance, encapsulation, abstraction, polymorphism, overloading, and overriding.",
    keyPoints: [
      "OOP is introduced with real-world objects that have properties and behavior, then mapped to classes and objects in programming.",
      "Classes, methods/functions, constructors, and the this keyword are explained with examples.",
      "Core interview concepts are revised: inheritance, encapsulation, abstraction, polymorphism, overloading, and overriding."
    ]
  },
  {
    label: "DBMS",
    threshold: 2,
    terms: [
      "database",
      "dbms",
      "table",
      "relation",
      "schema",
      "query",
      "sql",
      "join",
      "stored procedure",
      "relational database",
      "nosql",
      "normalization",
      "index",
      "transaction"
    ],
    summary:
      "the final major part revises DBMS basics: how users interact with data through GUI/CLI and database layers, tables and relationships, keys, SQL queries, joins, stored procedures, relational databases, NoSQL, scaling limits, and common interview points.",
    keyPoints: [
      "DBMS is explained as the system that stores, organizes, and retrieves application data through layers between the user interface and storage.",
      "The lecture revises tables, relationships, course/user examples, keys, SQL queries, joins, and stored procedures.",
      "Relational databases and NoSQL are compared through structure, scaling, and performance tradeoffs."
    ]
  }
];

const SUMMARY_LANGUAGES = {
  english: {
    id: "english",
    label: "English",
    promptName: "English",
    translateCode: "en"
  },
  hindi: {
    id: "hindi",
    label: "Hindi",
    promptName: "Hindi",
    translateCode: "hi"
  }
};
