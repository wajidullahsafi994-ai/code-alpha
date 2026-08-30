"""
app.py
------
Flask web server for the FAQ Chatbot.

Routes:
  GET  /            → renders the chat UI
  POST /chat        → accepts JSON {message}, returns JSON response
  POST /clear       → clears the session's conversation history
  GET  /history     → returns the full conversation history as JSON
  GET  /faqs        → returns all FAQ questions (for the suggestion panel)
"""

from flask import Flask, render_template, request, jsonify, session
from chatbot import Chatbot
import os
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "faq-chatbot-secret-2024")

# ── per-session chatbot instances ──────────────────────────────────────────────
# Keyed by session ID; each browser tab gets its own Chatbot instance.
_bots: dict[str, Chatbot] = {}


def _get_bot() -> Chatbot:
    """Return (or create) the Chatbot instance for the current session."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]
    if sid not in _bots:
        _bots[sid] = Chatbot()
    return _bots[sid]


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accept a user message and return the chatbot reply.

    Request body  (JSON): { "message": "<user text>" }
    Response body (JSON): {
        "answer"    : str,
        "matched_q" : str | null,
        "score"     : float,
        "confident" : bool,
        "intent"    : str,
        "timestamp" : str
    }
    """
    data        = request.get_json(force=True, silent=True) or {}
    user_input  = str(data.get("message", "")).strip()

    bot      = _get_bot()
    response = bot.respond(user_input)
    return jsonify(response)


@app.route("/clear", methods=["POST"])
def clear():
    """Clear the current session's conversation history."""
    bot = _get_bot()
    bot.clear_history()
    return jsonify({"status": "cleared"})


@app.route("/history", methods=["GET"])
def history():
    """Return the full conversation history for the current session."""
    bot = _get_bot()
    return jsonify(bot.get_history())


@app.route("/faqs", methods=["GET"])
def faqs():
    """Return all FAQ questions for the suggestion panel."""
    from matcher import FAQMatcher
    # Re-use the already-loaded singleton via the chatbot module
    import chatbot as _cb
    questions = [faq["question"] for faq in _cb._matcher.faqs]
    return jsonify(questions)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
