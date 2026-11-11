"""Asynchronous OpenAI SDK answer worker for the Step 33 caption application.

Compatible with: AnswerWorker(on_result, profile=...), .submit(), .close().
The callback is invoked on the worker thread; enqueue UI events, do not touch Tkinter.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import traceback

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    APIStatusError,
)

# Load project .env even when the app is launched from a different directory.
load_dotenv(Path(__file__).resolve().parent / '.env')

DEFAULT_MODEL = 'gpt-4.1-mini'
SYSTEM_PROMPT = (
    'You help someone practice software engineering interviews. '
    'Write a natural, first-person, conversational spoken answer, around 50-90 words. '
    'Be direct and specific. Avoid generic filler, buzzwords, and bullet points. '
    "Do not invent the candidate's employment history, projects, or metrics. "
    'If a question asks about personal experience not provided, supply an adaptable '
    'example structure and explicitly mark details to personalize. '
    'For technical questions, give a concise accurate explanation. '
    'Avoid claiming knowledge beyond the supplied profile.'
)


def generate_answer(
    question: str,
    profile: str = '',
    *,
    model: str | None = None,
    timeout: float = 20.0,
) -> str:
    """Generate one concise answer using the official OpenAI Python SDK.

    Does not expose secrets in error messages. Timeout and API processing time
    are controlled independently of the application's GPU transcription loop.
    """
    question = ' '.join(question.split())
    if not question:
        raise ValueError('Question cannot be empty.')

    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is missing. Add it to .env and restart the app.')

    chosen_model = (model or os.getenv('OPENAI_MODEL') or DEFAULT_MODEL).strip()
    print(f'[ANSWER API] model={chosen_model} | question_chars={len(question)}', flush=True)
    user_content = (
        f'Candidate background (use only these facts):\n{profile[:12000] or "Not provided"}\n\n'
        f'Interviewer question:\n{question}'
    )
    try:
        client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        print('[ANSWER API] Request sent', flush=True)
        response = client.chat.completions.create(
            model=chosen_model,
            reasoning_effort=os.getenv('OPENAI_REASONING_EFFORT') or "low",
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            max_completion_tokens=220,
        )
    except AuthenticationError as exc:
        raise RuntimeError('OpenAI authentication failed (401). Check OPENAI_API_KEY.') from exc
    except PermissionDeniedError as exc:
        raise RuntimeError('OpenAI access denied (403). Check project permissions.') from exc
    except NotFoundError as exc:
        raise RuntimeError(f'OpenAI model not available (404): {chosen_model}.') from exc
    except RateLimitError as exc:
        raise RuntimeError('OpenAI rate limit or API quota reached (429).') from exc
    except APITimeoutError as exc:
        raise RuntimeError(f'OpenAI request timed out after {timeout:g} seconds.') from exc
    except APIConnectionError as exc:
        raise RuntimeError('Unable to connect to the OpenAI API. Check your connection.') from exc
    except BadRequestError as exc:
        raise RuntimeError('OpenAI rejected the request (400). Check the selected model and parameters.') from exc
    except APIStatusError as exc:
        raise RuntimeError(f'OpenAI API returned HTTP {exc.status_code}.') from exc

    if not response.choices:
        raise RuntimeError('OpenAI returned no answer choices.')
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        reason = response.choices[0].finish_reason
        raise RuntimeError(f'OpenAI returned no answer text (finish_reason={reason}).')
    print(f'[ANSWER API] Response received | chars={len(answer.strip())}', flush=True)
    return answer.strip()


def stream_answer(question: str, profile: str = "", *, model: str | None = None,
                  timeout: float = 30.0):
    """Yield text chunks as OpenAI returns them. Called only from background worker.

    The supported reasoning_effort values vary by model. Set
    OPENAI_REASONING_EFFORT=low in .env, or omit it for SDK defaults.
    """
    question = " ".join(question.split())
    if not question:
        raise ValueError("Question cannot be empty.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    chosen_model = (model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
    max_tokens = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "320"))
    user_content = (
        f"Candidate background (only these facts):\n{profile[:4000] or 'Not provided'}\n\n"
        f"Question:\n{question}"
    )
    kwargs = dict(
        model=chosen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=max_tokens,
        stream=True,
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    print(f"[ANSWER API] Streaming request model={chosen_model} effort={reasoning_effort or 'default'}", flush=True)
    client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
    try:
        stream = client.chat.completions.create(**kwargs)
        try:
            for response_chunk in stream:
                if not response_chunk.choices:
                    continue
                delta = response_chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    yield content
        finally:
            # Releases the HTTP response even if the question was superseded.
            if hasattr(stream, "close"):
                stream.close()
    except AuthenticationError as exc:
        raise RuntimeError("OpenAI authentication failed (401).") from exc
    except PermissionDeniedError as exc:
        raise RuntimeError("OpenAI access denied (403).") from exc
    except NotFoundError as exc:
        raise RuntimeError(f"OpenAI model not available (404): {chosen_model}.") from exc
    except RateLimitError as exc:
        raise RuntimeError("OpenAI rate limit or quota reached (429).") from exc
    except APITimeoutError as exc:
        raise RuntimeError("OpenAI request timed out.") from exc
    except APIConnectionError as exc:
        raise RuntimeError("Unable to connect to OpenAI API.") from exc
    except BadRequestError as exc:
        raise RuntimeError(
            "OpenAI rejected parameters (400). Verify model supports "
            "OPENAI_REASONING_EFFORT, or remove this variable from .env."
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(f"OpenAI API returned HTTP {exc.status_code}.") from exc


class AnswerWorker:
    """One background API worker, latest question wins, stream-safe callbacks.

    on_chunk(seq, chunk, elapsed): UI queue ONLY. Called from worker thread.
    on_result(seq, status, answer, elapsed): UI queue ONLY.

    Existing tests can inject `generate` to exercise legacy nonstreaming path.
    """

    def __init__(self, on_result, *, profile='', generate=None,
                 on_chunk=None, stream_generate=None):
        self.on_result = on_result
        self.on_chunk = on_chunk
        self.profile = profile
        self.generate = generate
        self.stream_generate = stream_generate or stream_answer
        self._lock = threading.Condition()
        self._pending = None
        self._generation = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name='AnswerAPI')
        self._thread.start()

    def submit(self, question: str):
        question = ' '.join(question.split())
        if not question:
            return None
        with self._lock:
            if self._stopped:
                return None
            self._generation += 1
            seq = self._generation
            self._pending = (seq, question)
            self._lock.notify_all()
            print(f'[ANSWER SUBMIT #{seq}] {question[:120]}', flush=True)
            return seq

    def _current(self, seq):
        with self._lock:
            return not self._stopped and self._generation == seq

    def _loop(self):
        while True:
            with self._lock:
                while not self._stopped and self._pending is None:
                    self._lock.wait()
                if self._stopped:
                    return
                seq, question = self._pending
                self._pending = None
            print(f'[ANSWER START #{seq}]', flush=True)
            started = time.perf_counter()
            pieces = []
            status = 'ok'
            first_text = None
            error_message = None
            try:
                if self.generate is not None:
                    # Compatibility for old sync tests and custom generators.
                    generated = self.generate(question, self.profile)
                    chunks = (generated,)
                else:
                    chunks = self.stream_generate(question, self.profile)
                for chunk in chunks:
                    if not self._current(seq):
                        break
                    if not chunk:
                        continue
                    pieces.append(chunk)
                    elapsed = time.perf_counter() - started
                    if first_text is None:
                        first_text = elapsed
                        print(f'[ANSWER FIRST TEXT #{seq}] {first_text:.3f}s', flush=True)
                    if self.on_chunk is not None:
                        try:
                            self.on_chunk(seq, chunk, elapsed)
                        except Exception:
                            print(f'[ANSWER CHUNK CALLBACK ERROR #{seq}]', flush=True)
                            traceback.print_exc()
            except Exception as exc:
                status = 'error'
                error_message = f'{type(exc).__name__}: {exc}'
                print(f'[ANSWER ERROR #{seq}] {error_message}', flush=True)
                traceback.print_exc()
            elapsed = time.perf_counter() - started
            if not self._current(seq):
                print(f'[ANSWER DISCARD #{seq}] Superseded or shutdown', flush=True)
                continue
            answer = ''.join(pieces)
            if status == 'ok' and not answer.strip():
                status = 'error'
                error_message = 'OpenAI returned no visible answer text. Check output token budget.'
            if status == 'error':
                answer = (answer + '\n\n' if answer else '') + (error_message or 'Unknown error')
            print(f'[ANSWER DONE #{seq}] status={status} time={elapsed:.2f}s', flush=True)
            try:
                self.on_result(seq, status, answer, elapsed)
                print(f'[ANSWER CALLBACK #{seq}] Delivered to UI event queue', flush=True)
            except Exception:
                print(f'[ANSWER CALLBACK ERROR #{seq}]', flush=True)
                traceback.print_exc()

    def close(self):
        # Only call on shutdown, not when the application first starts.
        with self._lock:
            self._stopped = True
            self._pending = None
            self._lock.notify_all()
