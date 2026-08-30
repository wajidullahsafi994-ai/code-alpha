"""
chatbot.py
----------
Core chatbot logic layer.

Sits between the web layer (app.py) and the matching engine (matcher.py).
Handles:
  - Greeting / farewell / thank-you intent detection (rule-based fast path)
  - Delegating everything else to FAQMatcher
  - Maintaining a simple in-memory conversation history per session
  - Returning structured response dicts consumed by the Flask routes
"""

import re
from datetime import datetime
from matcher import FAQMatcher

# ── singleton matcher (loaded once at import time) ─────────────────────────────
_matcher = FAQMatcher()

# ── simple intent patterns (order matters — checked top to bottom) ─────────────
_INTENT_PATTERNS = [
    # (compiled regex, response string)
    (
        re.compile(
            r"\b(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening))\b",
            re.IGNORECASE,
        ),
        "Hello! 👋 I'm your FAQ assistant. Ask me anything about orders, "
        "shipping, returns, payments, or your account.",
    ),
    (
        re.compile(
            r"\b(bye|goodbye|see\s*you|take\s*care|farewell|exit|quit)\b",
            re.IGNORECASE,
        ),
        "Goodbye! 👋 Feel free to come back if you have more questions.",
    ),
    (
        re.compile(
            r"\b(thank(s| you)|thx|appreciate|cheers)\b",
            re.IGNORECASE,
        ),
        "You're welcome! 😊 Is there anything else I can help you with?",
    ),
    (
        re.compile(
            r"\b(help|what can you do|how does this work|what do you know)\b",
            re.IGNORECASE,
        ),
        "I can answer questions about:\n"
        "• Orders & tracking\n"
        "• Returns & exchanges\n"
        "• Shipping & delivery\n"
        "• Payments & discounts\n"
        "• Account & password\n"
        "• Warranty & loyalty rewards\n\n"
        "Just type your question and I'll find the best answer!",
    ),
]


class Chatbot:
    """
    Stateful chatbot that tracks conversation history and resolves user
    messages to appropriate responses.
    """

    def __init__(self):
        # List of {"role": "user"|"bot", "text": str, "timestamp": str}
        self.history: list[dict] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def respond(self, user_input: str) -> dict:
        """
        Process *user_input* and return a response dict:
        {
            "answer"    : str,
            "matched_q" : str | None,   # the FAQ question that was matched
            "score"     : float,
            "confident" : bool,
            "intent"    : str,          # "greeting"|"farewell"|"thanks"|
                                        #  "help"|"faq"|"fallback"
            "timestamp" : str,
        }
        """
        user_input = user_input.strip()
        timestamp  = datetime.now().strftime("%H:%M")

        # Record user turn
        self._record("user", user_input, timestamp)

        # 1. Empty input guard
        if not user_input:
            reply = self._build_reply(
                answer    = "Please type a question and I'll do my best to help!",
                matched_q = None,
                score     = 0.0,
                confident = False,
                intent    = "empty",
                timestamp = timestamp,
            )
            self._record("bot", reply["answer"], timestamp)
            return reply

        # 2. Rule-based intent fast path
        for pattern, response in _INTENT_PATTERNS:
            if pattern.search(user_input):
                intent = self._detect_intent_name(pattern)
                reply  = self._build_reply(
                    answer    = response,
                    matched_q = None,
                    score     = 1.0,
                    confident = True,
                    intent    = intent,
                    timestamp = timestamp,
                )
                self._record("bot", reply["answer"], timestamp)
                return reply

        # 3. FAQ matching
        result = _matcher.match(user_input)
        intent = "faq" if result["confident"] else "fallback"

        reply = self._build_reply(
            answer    = result["answer"],
            matched_q = result["question"],
            score     = result["score"],
            confident = result["confident"],
            intent    = intent,
            timestamp = timestamp,
        )
        self._record("bot", reply["answer"], timestamp)
        return reply

    def get_history(self) -> list[dict]:
        """Return a copy of the full conversation history."""
        return list(self.history)

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history.clear()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _record(self, role: str, text: str, timestamp: str) -> None:
        self.history.append({"role": role, "text": text, "timestamp": timestamp})

    @staticmethod
    def _build_reply(
        answer: str,
        matched_q,
        score: float,
        confident: bool,
        intent: str,
        timestamp: str,
    ) -> dict:
        return {
            "answer"    : answer,
            "matched_q" : matched_q,
            "score"     : score,
            "confident" : confident,
            "intent"    : intent,
            "timestamp" : timestamp,
        }

    @staticmethod
    def _detect_intent_name(pattern: re.Pattern) -> str:
        src = pattern.pattern
        if "hi|hello" in src:
            return "greeting"
        if "bye|goodbye" in src:
            return "farewell"
        if "thank" in src:
            return "thanks"
        if "help" in src:
            return "help"
        return "rule_based"


# ── CLI fallback (run without the web server) ──────────────────────────────────
if __name__ == "__main__":
    bot = Chatbot()
    print("FAQ Chatbot (type 'quit' to exit)\n" + "=" * 40)
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_msg.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye! 👋")
            break

        response = bot.respond(user_msg)
        print(f"Bot: {response['answer']}")
        if response["matched_q"]:
            print(f"     [Matched: \"{response['matched_q']}\" | score={response['score']}]")
        print()
