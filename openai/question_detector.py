"""Step 32: deterministic interview-turn/question detection (no network or LLM).

Consume final transcript checkpoints, provisional LIVE updates, and a VAD speech-end
signal. Only FINAL can trigger automatic detection, and only after VAD ends.
No checkpoint boundary is treated as a question end.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import time

MIN_WORDS = 4
MAX_QUESTION_WORDS = 100
QUESTION_START = re.compile(
    r"^(?:so[,.]?\s+|and[,.]?\s+|okay[,.]?\s+|well[,.]?\s+)*"
    r"(?:what|why|how|when|where|who|which|whose|whom|"
    r"do|does|did|can|could|would|will|should|is|are|was|were|"
    r"have|has|had|may|might|tell me|walk me|talk me|explain|describe|"
    r"give me|share|outline|compare|discuss|help me understand)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionEvent:
    phrase_id: int
    text: str
    source: str  # automatic | manual
    reason: str


class QuestionDetector:
    def __init__(self, min_words=MIN_WORDS):
        self.min_words = min_words
        self.phrases = {}
        self.emitted_auto = set()
        self.last_manual = None

    def start(self, phrase_id):
        self.phrases[phrase_id] = {"finals": [], "live": "", "vad_ended": False, "completed": False}
        # Retain limited history; IDs are unique during a capture run.
        for key in list(self.phrases):
            if key != phrase_id and len(self.phrases) > 12 and self.phrases[key]["completed"]:
                del self.phrases[key]

    def live(self, phrase_id, text):
        if phrase_id in self.phrases and not self.phrases[phrase_id]["completed"]:
            self.phrases[phrase_id]["live"] = text.strip()

    def checkpoint(self, phrase_id, text, last=False):
        state = self.phrases.get(phrase_id)
        if state is None or state["completed"]:
            return None
        if text.strip():
            state["finals"].append(text.strip())
            state["live"] = ""  # A new checkpoint has superseded that hypothesis.
        if last:
            state["completed"] = True
            return self._maybe_auto(phrase_id)
        return None

    def vad_end(self, phrase_id):
        state = self.phrases.get(phrase_id)
        if state is None:
            return None
        state["vad_ended"] = True
        return self._maybe_auto(phrase_id)

    def _maybe_auto(self, phrase_id):
        state = self.phrases.get(phrase_id)
        if (not state or not state["vad_ended"] or not state["completed"]
                or phrase_id in self.emitted_auto):
            return None
        text = self.extract_question(" ".join(state["finals"]))
        if not text:
            return None
        self.emitted_auto.add(phrase_id)
        return QuestionEvent(phrase_id, text, "automatic", "VAD ended + FINAL complete + question-like text")

    def manual(self, phrase_id, now=None):
        state = self.phrases.get(phrase_id)
        if not state:
            return None
        text = " ".join(state["finals"] + ([state["live"]] if state["live"] else [])).strip()
        if not text:
            return None
        text = " ".join(text.split()[-MAX_QUESTION_WORDS:])
        now = time.monotonic() if now is None else now
        signature = (phrase_id, text)
        if self.last_manual and self.last_manual[0] == signature and now - self.last_manual[1] < 2:
            return None
        self.last_manual = (signature, now)
        return QuestionEvent(phrase_id, text, "manual", "User clicked Detect")

    def extract_question(self, text):
        text = " ".join(text.strip().split())
        if not text:
            return None
        # Choose last question/sentence, since interviewer can offer explanatory setup.
        # Do not interpret a period inside an abbreviation as decisive; this is heuristic.
        candidates = re.split(r"(?<=[.!?])\s+", text)
        for candidate in reversed(candidates):
            candidate = candidate.strip()
            if not candidate:
                continue
            words = candidate.split()
            if len(words) < self.min_words:
                continue
            if len(words) > MAX_QUESTION_WORDS:
                candidate = " ".join(words[-MAX_QUESTION_WORDS:])
            if candidate.endswith("?") or QUESTION_START.match(candidate):
                return candidate
        return None
