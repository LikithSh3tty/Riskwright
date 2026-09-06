"""Server-side conversation memory.

Keyed by a session id the client sends. Deliberately not in the client's
session state: if conversation history lived in the UI, replacing the frontend
would mean reimplementing memory, and the API would be unable to answer a
follow-up question on its own.

Only the question and the SQL that answered it are retained, never the result
rows. Rows are the expensive part of a transcript and contribute nothing to
generating the next query. A turn costs roughly 40 tokens to replay instead of
several hundred.

Storage is an in-process dictionary with a TTL. That is honest for a
single-container deployment and is named as a limitation in the README: it does
not survive a restart and does not scale horizontally. Redis is the obvious
next step and is not warranted here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from src.utils.logger import get_logger

log = get_logger(__name__)

# Turns replayed into the prompt. Six is enough for "and for X?" chains without
# letting an old topic drag a new question off course.
MAX_TURNS = 6

# Sessions idle longer than this are dropped.
TTL_SECONDS = 60 * 60

# Ceiling on tracked sessions, so a crawler cannot grow the process unbounded.
MAX_SESSIONS = 500


@dataclass
class Turn:
    question: str
    action: str
    sql: str | None = None
    note: str | None = None

    def render(self) -> list[dict]:
        """Replay as a message pair.

        The assistant side is the SQL, or the reason it declined. Replaying a
        refusal matters: without it the model re-attempts a question it has
        already correctly refused.
        """
        if self.action == "sql" and self.sql:
            assistant = f"SQL: {self.sql}"
        else:
            assistant = f"[{self.action}] {self.note or ''}".strip()
        return [
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": assistant},
        ]


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


class ConversationMemory:
    def __init__(self, max_turns: int = MAX_TURNS, ttl: int = TTL_SECONDS):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_turns = max_turns
        self._ttl = ttl

    def _expire(self, now: float) -> None:
        stale = [k for k, s in self._sessions.items() if now - s.last_seen > self._ttl]
        for key in stale:
            del self._sessions[key]
        if stale:
            log.info("expired %d idle chat session(s)", len(stale))

        if len(self._sessions) > MAX_SESSIONS:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].last_seen)
            for key, _ in oldest[: len(self._sessions) - MAX_SESSIONS]:
                del self._sessions[key]

    def history(self, session_id: str) -> list[dict]:
        """Prior turns as Anthropic message dicts, oldest first."""
        now = time.time()
        with self._lock:
            self._expire(now)
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_seen = now
            turns = session.turns[-self._max_turns:]

        messages: list[dict] = []
        for turn in turns:
            messages.extend(turn.render())
        return messages

    def record(
        self,
        session_id: str,
        question: str,
        action: str,
        sql: str | None = None,
        note: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._expire(now)
            session = self._sessions.setdefault(session_id, Session())
            session.turns.append(Turn(question, action, sql, note))
            # Trim eagerly so a long session does not grow without bound. Twice
            # the replay window is kept so turn_count stays meaningful.
            limit = self._max_turns * 2
            if len(session.turns) > limit:
                session.turns = session.turns[-limit:]
            session.last_seen = now

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def turn_count(self, session_id: str) -> int:
        with self._lock:
            session = self._sessions.get(session_id)
            return len(session.turns) if session else 0

    def active_sessions(self) -> int:
        with self._lock:
            self._expire(time.time())
            return len(self._sessions)


# One store per API process.
MEMORY = ConversationMemory()
