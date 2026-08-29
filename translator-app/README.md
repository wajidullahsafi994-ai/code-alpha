# Text Translator App

A simple full-stack translator: pick a source and target language, enter text,
and get a translation powered by the **MyMemory Translation API** — free,
no signup, and no API key required.

## Features
- Source & target language selection
- Live translation via MyMemory API
- Copy-to-clipboard button
- Text-to-speech playback of the translated text
- Swap languages button

## Project structure
```
translator-app/
├── client/          # plain HTML/CSS/JS frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── server/          # Node/Express backend (proxies requests to MyMemory)
│   ├── server.js
│   ├── package.json
│   └── .env.example
├── .gitignore
└── README.md
```

## 1. Install backend dependencies
No API key or signup needed — MyMemory is free and open.
```bash
cd server
npm install
```
(Optional) copy `.env.example` to `.env` and add your email to raise the
free daily limit from 5,000 to 50,000 words/day — totally optional, the
app works fine without it.

## 2. Run the backend
```bash
npm start
```
This starts the API at `http://localhost:5000`.

## 3. Run the frontend
Simplest option — open `client/index.html` directly in your browser,
or serve it with a local server (e.g. VS Code "Live Server" extension)
so it behaves like a normal web page.

## 4. Test it
Type text, choose languages, click **Translate**. Try Copy and Listen too.

## Notes for submission
- Note in your submission that the app uses the MyMemory Translation API
  (free, keyless) as the translation provider, since it satisfies the task's
  "use a translation API" requirement without needing a paid account.
- The "Detect language" option defaults to English as the source, since
  MyMemory doesn't support real language auto-detection.
- If you want to deploy this (e.g. for the internship demo), host the
  `server/` folder somewhere like Render or Railway, and update
  `BACKEND_URL` in `client/script.js` to point to that live URL.