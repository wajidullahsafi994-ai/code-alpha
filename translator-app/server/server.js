// server.js
// Simple Express backend that proxies translation requests to the
// MyMemory Translation API — free, no signup, no API key required.

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 5000;

const MYMEMORY_ENDPOINT = "https://api.mymemory.translated.net/get";
// Optional: add your email to raise MyMemory's free daily limit
// from 5,000 to 50,000 words/day. Set MYMEMORY_EMAIL in .env if you want this.
const MYMEMORY_EMAIL = process.env.MYMEMORY_EMAIL || "";

// A fixed list of common languages, since MyMemory has no "list languages" endpoint.
// code: display name (used to populate the dropdowns on the frontend)
const SUPPORTED_LANGUAGES = {
  en: "English",
  ur: "Urdu",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  ar: "Arabic",
  hi: "Hindi",
  "zh-CN": "Chinese (Simplified)",
  ja: "Japanese",
  ko: "Korean",
  ru: "Russian",
  tr: "Turkish",
  fa: "Persian",
  bn: "Bengali",
  pa: "Punjabi",
};

app.use(cors());
app.use(express.json());

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

// GET /api/languages -> list of supported languages (code + name)
app.get("/api/languages", (req, res) => {
  res.json(SUPPORTED_LANGUAGES);
});

// POST /api/translate -> { text, from, to } -> { translatedText }
app.post("/api/translate", async (req, res) => {
  const { text, from, to } = req.body;

  if (!text || !to) {
    return res.status(400).json({ error: "Fields 'text' and 'to' are required" });
  }

  // MyMemory needs an explicit source language; default to English if "auto" was picked,
  // since it doesn't support real auto-detection.
  const sourceLang = from && from !== "auto" ? from : "en";

  try {
    const response = await axios.get(MYMEMORY_ENDPOINT, {
      params: {
        q: text,
        langpair: `${sourceLang}|${to}`,
        de: MYMEMORY_EMAIL || undefined,
      },
    });

    const data = response.data;
    if (!data.responseData || data.responseStatus !== 200) {
      throw new Error(data.responseDetails || "Translation failed");
    }

    res.json({
      translatedText: data.responseData.translatedText,
      detectedLanguage: sourceLang,
    });
  } catch (err) {
    console.error("Translation error:", err.response ? err.response.data : err.message);
    res.status(500).json({ error: "Translation failed. Please try again." });
  }
});

app.listen(PORT, () => {
  console.log(`Translator backend running on http://localhost:${PORT}`);
});