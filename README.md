# LectureLens AI

A Node + Express web app that summarizes YouTube lectures from transcripts using Gemini.

## Features

- Paste a YouTube URL or video ID
- Extract transcript with `youtube-transcript`
- Chunk long transcripts before sending them to the model
- Return a structured summary, key points, and timestamped notes
- Choose summary language from the UI
- Fall back to a local transcript-based summary if the AI provider is overloaded
- Serve a simple HTML/CSS/JS frontend from Express

## Setup

```bash
npm install
copy .env.example .env
```

Add your Groq API key to `.env`:

```bash
AI_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
LOCAL_SUMMARY_FALLBACK=true
PORT=3000
```

Run the app:

```bash
npm start
```

Open `http://localhost:3000`.

## API

```http
POST /api/summarize
Content-Type: application/json

{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

Response:

```json
{
  "videoId": "VIDEO_ID",
  "summary": "Clear lecture summary...",
  "keyPoints": ["Main idea", "Important detail"],
  "timestampedNotes": [
    {
      "time": "02:15",
      "seconds": 135,
      "note": "Important explanation starts here."
    }
  ]
}
```
