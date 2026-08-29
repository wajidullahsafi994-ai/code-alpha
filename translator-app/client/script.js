// script.js
// Talks to our own backend at BACKEND_URL, which in turn calls MyMemory Translation API.

const BACKEND_URL = "http://localhost:5000";

const fromLangSelect = document.getElementById("fromLang");
const toLangSelect = document.getElementById("toLang");
const sourceText = document.getElementById("sourceText");
const translatedText = document.getElementById("translatedText");
const translateBtn = document.getElementById("translateBtn");
const copyBtn = document.getElementById("copyBtn");
const speakBtn = document.getElementById("speakBtn");
const swapBtn = document.getElementById("swapBtn");
const statusMsg = document.getElementById("statusMsg");
const charCount = document.getElementById("charCount");

// A small fallback list in case the backend/languages endpoint is unreachable
const FALLBACK_LANGUAGES = {
  en: "English",
  ur: "Urdu",
  es: "Spanish",
  fr: "French",
  de: "German",
  ar: "Arabic",
  hi: "Hindi",
  "zh-CN": "Chinese (Simplified)",
  ja: "Japanese",
  ru: "Russian",
};

function populateSelect(select, languages, includeAuto) {
  select.innerHTML = "";
  if (includeAuto) {
    const autoOpt = document.createElement("option");
    autoOpt.value = "auto";
    autoOpt.textContent = "Detect language";
    select.appendChild(autoOpt);
  }
  Object.entries(languages).forEach(([code, info]) => {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = typeof info === "string" ? info : info.name;
    select.appendChild(opt);
  });
}

async function loadLanguages() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/languages`);
    if (!res.ok) throw new Error("Failed to load languages");
    const data = await res.json();
    populateSelect(fromLangSelect, data, true);
    populateSelect(toLangSelect, data, false);
    fromLangSelect.value = "auto";
    toLangSelect.value = "es" in data ? "es" : Object.keys(data)[0];
  } catch (err) {
    console.warn("Falling back to default language list:", err.message);
    populateSelect(fromLangSelect, FALLBACK_LANGUAGES, true);
    populateSelect(toLangSelect, FALLBACK_LANGUAGES, false);
    fromLangSelect.value = "auto";
    toLangSelect.value = "es";
  }
}

function setStatus(message, type) {
  statusMsg.textContent = message || "";
  statusMsg.className = "status-msg" + (type ? ` ${type}` : "");
}

async function translateText() {
  const text = sourceText.value.trim();
  if (!text) {
    setStatus("Please enter some text to translate.", "error");
    return;
  }

  translateBtn.disabled = true;
  translateBtn.textContent = "Translating...";
  setStatus("");

  try {
    const res = await fetch(`${BACKEND_URL}/api/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        from: fromLangSelect.value,
        to: toLangSelect.value,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Translation failed");

    translatedText.value = data.translatedText;
    setStatus("Translated successfully.", "success");
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    translateBtn.disabled = false;
    translateBtn.textContent = "Translate";
  }
}

function copyTranslation() {
  if (!translatedText.value) return;
  navigator.clipboard
    .writeText(translatedText.value)
    .then(() => setStatus("Copied to clipboard.", "success"))
    .catch(() => setStatus("Could not copy text.", "error"));
}

function speakTranslation() {
  if (!translatedText.value) return;
  if (!("speechSynthesis" in window)) {
    setStatus("Text-to-speech is not supported in this browser.", "error");
    return;
  }
  const utterance = new SpeechSynthesisUtterance(translatedText.value);
  utterance.lang = toLangSelect.value;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function swapLanguages() {
  if (fromLangSelect.value === "auto") {
    setStatus("Can't swap while source is set to 'Detect language'.", "error");
    return;
  }
  const temp = fromLangSelect.value;
  fromLangSelect.value = toLangSelect.value;
  toLangSelect.value = temp;

  const tempText = sourceText.value;
  sourceText.value = translatedText.value;
  translatedText.value = tempText;
  charCount.textContent = sourceText.value.length;
}

sourceText.addEventListener("input", () => {
  charCount.textContent = sourceText.value.length;
});

translateBtn.addEventListener("click", translateText);
copyBtn.addEventListener("click", copyTranslation);
speakBtn.addEventListener("click", speakTranslation);
swapBtn.addEventListener("click", swapLanguages);

sourceText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    translateText();
  }
});

loadLanguages();