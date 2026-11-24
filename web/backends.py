"""Agent backends for the adversarial reasoning web UI.

Each backend turns a prompt into a text reply. Real backends shell out to the
same CLIs used by ``adversarial_review.sh`` (prompt on stdin, list argv, never
``shell=True``); ``MockBackend`` is deterministic and works offline.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import re
import urllib.request
from dataclasses import dataclass
from typing import IO, Callable, Dict, Iterable, List, Optional, Tuple

from api_keys import ApiKeyStore

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 600
STDERR_TAIL_CHARS = 800

TextSink = Callable[[str], None]  # receives each new piece of a reply while it is being generated


class BackendError(RuntimeError):
    """The agent CLI failed (non-zero exit, missing binary, empty output)."""


class BackendTimeout(BackendError):
    """The agent CLI did not answer within the configured timeout."""


@dataclass(frozen=True)
class ImageFile:
    """An image sent with a chat reply. ``number`` is its "[image N]" label in the history text."""

    number: int
    path: str
    media_type: str


@dataclass(frozen=True)
class TurnRequest:
    """Everything a backend needs to produce one debate turn."""

    role: str  # "proposer" | "critic" | "judge" | "chat"
    round: int
    prompt: str
    model: Optional[str] = None   # None: the backend's default (its --model flag or the CLI's own)
    effort: Optional[str] = None  # None: the CLI's default reasoning effort
    images: Tuple[ImageFile, ...] = ()  # chat only; only sent to backends with supports_images


class Backend:
    name = "base"
    # Values the UI may pick per seat. They end up in a CLI argv or an API body, so anything outside
    # these tuples is rejected by DebateConfig.validate. Empty tuple: no choice offered.
    models: Tuple[str, ...] = ()
    efforts: Tuple[str, ...] = ()
    supports_images = False  # the app refuses images for an agent that would silently drop them

    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, request: TurnRequest) -> str:
        raise NotImplementedError

    def stream(self, request: TurnRequest, on_text: TextSink) -> str:
        """Like ``complete``, and also hand each new piece of the reply to ``on_text`` as it is
        generated. Returns the full reply. Backends that cannot stream answer in one block and
        never call ``on_text``."""
        return self.complete(request)


def _tail(text: str) -> str:
    return text[-STDERR_TAIL_CHARS:].strip()


def _run_cli(argv: List[str], prompt: str, cwd: str, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # child is killed by subprocess.run
        raise BackendTimeout(f"{argv[0]} timed out after {timeout:.0f}s") from exc
    except FileNotFoundError as exc:
        raise BackendError(f"{argv[0]} not found on PATH") from exc


# Pure reasoning: empty cwd, no user/global CLAUDE.md or hooks (they would leak the operator's
# own instructions into the debate), no tools, no MCP servers.
CLAUDE_ISOLATION_FLAGS = ["--setting-sources", "project", "--tools", "", "--strict-mcp-config"]
# Same idea for gemini: plan mode is read-only (no file edits, no shell), no extensions, and the
# empty scratch dir is trusted so headless mode does not stop on the folder-trust prompt.
GEMINI_ISOLATION_FLAGS = ["--approval-mode", "plan", "-e", "none", "--skip-trust"]
# One JSON event per line, including partial text (stream_event / content_block_delta / text_delta)
# and a final ``result`` event. --verbose is required by the CLI for stream-json with --print.
CLAUDE_STREAM_FLAGS = ["--output-format", "stream-json", "--include-partial-messages", "--verbose"]


@dataclass
class ClaudeStreamResult:
    streamed: str           # concatenated text deltas
    result: Optional[str]   # text of the final ``result`` event (authoritative), None if it never came
    is_error: bool


def parse_claude_stream(lines: Iterable[str], on_text: TextSink) -> ClaudeStreamResult:
    """Read ``claude --output-format stream-json`` lines, forwarding text deltas to ``on_text``.
    Non-JSON lines and other event types (thinking, rate limits, tool use) are skipped."""
    pieces: List[str] = []
    result: Optional[str] = None
    is_error = False
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "stream_event":
            inner = event.get("event") or {}
            delta = inner.get("delta") or {}
            if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta" and delta.get("text"):
                pieces.append(delta["text"])
                on_text(delta["text"])
        elif event.get("type") == "result":
            result = event.get("result") if isinstance(event.get("result"), str) else None
            is_error = bool(event.get("is_error")) or event.get("subtype") != "success"
    return ClaudeStreamResult(streamed="".join(pieces), result=result, is_error=is_error)


class ClaudeBackend(Backend):
    """``claude --print`` isolated from user settings, tools and MCP, run in an empty scratch dir."""

    name = "claude"
    models = ("fable", "opus", "sonnet", "haiku")           # aliases for the latest of each family
    efforts = ("low", "medium", "high", "xhigh", "max")    # claude --effort, Claude Code 2.1.322
    supports_images = True  # stream() only: images go as content blocks with --input-format stream-json

    def __init__(self, model: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which("claude") is not None

    def build_argv(self, model: Optional[str] = None, effort: Optional[str] = None) -> List[str]:
        argv = ["claude", "--print", *CLAUDE_ISOLATION_FLAGS]
        model = model or self.model
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        return argv

    def complete(self, request: TurnRequest) -> str:
        if request.images:
            raise BackendError("claude takes images only in stream mode")  # never drop them silently
        with tempfile.TemporaryDirectory(prefix="advr-claude-") as scratch:
            started = time.monotonic()
            proc = _run_cli(self.build_argv(request.model, request.effort), request.prompt, scratch, self.timeout)
            logger.info("claude role=%s round=%d model=%s effort=%s exit=%d %.1fs", request.role, request.round,
                        request.model or self.model or "default", request.effort or "default",
                        proc.returncode, time.monotonic() - started)
        if proc.returncode != 0:
            raise BackendError(f"claude exited {proc.returncode}: {_tail(proc.stderr or proc.stdout)}")
        text = proc.stdout.strip()
        if not text:
            raise BackendError("claude returned empty output")
        return text

    def stream(self, request: TurnRequest, on_text: TextSink) -> str:
        argv = self.build_argv(request.model, request.effort) + CLAUDE_STREAM_FLAGS
        stdin_text = request.prompt
        if request.images:
            argv += ["--input-format", "stream-json"]
            stdin_text = claude_user_message(request)
        timed_out = threading.Event()
        with tempfile.TemporaryDirectory(prefix="advr-claude-") as scratch, tempfile.TemporaryFile("w+") as errf:
            started = time.monotonic()
            try:
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
                                        text=True, cwd=scratch)
            except FileNotFoundError as exc:
                raise BackendError("claude not found on PATH") from exc

            def kill_on_timeout() -> None:
                timed_out.set()
                proc.kill()  # closes stdout, so the read loop below ends

            timer = threading.Timer(self.timeout, kill_on_timeout)
            timer.start()
            try:
                _write_prompt(proc.stdin, stdin_text)
                parsed = parse_claude_stream(proc.stdout, on_text)
                proc.wait()
            finally:
                timer.cancel()
                if proc.poll() is None:  # on_text raised, or another error: never leave the CLI running
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
            errf.seek(0)
            stderr = errf.read()
        logger.info("claude role=%s round=%d model=%s effort=%s images=%d exit=%d stream %.1fs", request.role,
                    request.round, request.model or self.model or "default", request.effort or "default",
                    len(request.images), proc.returncode, time.monotonic() - started)
        if timed_out.is_set():
            raise BackendTimeout(f"claude timed out after {self.timeout:.0f}s")
        if proc.returncode != 0 or parsed.is_error:
            raise BackendError(f"claude exited {proc.returncode}: {_tail(parsed.result or stderr)}")
        text = (parsed.result if parsed.result is not None else parsed.streamed).strip()
        if not text:
            raise BackendError("claude returned empty output")
        return text


def claude_user_message(request: TurnRequest) -> str:
    """One stream-json input line: each image after its "Image N:" label, then the prompt text."""
    content: List[Dict] = []
    for image in request.images:
        with open(image.path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        content.append({"type": "text", "text": f"Image {image.number}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": image.media_type, "data": data}})
    content.append({"type": "text", "text": request.prompt})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def _write_prompt(stdin: IO[str], prompt: str) -> None:
    try:
        stdin.write(prompt)
        stdin.close()
    except BrokenPipeError:
        pass  # the CLI died before reading it all: its exit code and stderr say why


def codex_error_from_jsonl(stdout: str) -> Optional[str]:
    """Return the message of the last ``error`` / ``turn.failed`` event in ``codex exec --json`` output."""
    message = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.failed":
            message = (event.get("error") or {}).get("message", message)
        elif event.get("type") == "error":
            message = event.get("message", message)
    if message:
        try:  # the API error is often itself a JSON document
            inner = json.loads(message)
            message = inner.get("error", {}).get("message", message)
        except (json.JSONDecodeError, AttributeError):
            pass
    return message


class CodexBackend(Backend):
    """``codex exec --json`` in a read-only sandbox; the answer is read from ``-o`` (final message only)."""

    name = "codex"
    # No model choice: this server's ChatGPT-linked account only accepts the --codex-model it starts with.
    efforts = ("low", "medium", "high")  # model_reasoning_effort
    supports_images = True  # codex exec -i FILE, in order

    def __init__(self, model: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which("codex") is not None

    def build_argv(self, scratch: str, out_file: str, effort: Optional[str] = None,
                   images: Tuple[ImageFile, ...] = ()) -> List[str]:
        argv = ["codex", "exec", "--json"]
        if self.model:
            argv += ["-m", self.model]
        if effort:
            argv += ["-c", f"model_reasoning_effort={effort}"]
        argv += ["--sandbox", "read-only", "--skip-git-repo-check", "-C", scratch, "-o", out_file]
        for image in images:
            argv += ["-i", image.path]
        return argv + ["-"]

    @staticmethod
    def codex_prompt(request: TurnRequest) -> str:
        """codex has no per-image label: say which numbers the attached files are, in order."""
        if not request.images:
            return request.prompt
        numbers = ", ".join(f"image {image.number}" for image in request.images)
        return f"Attached images, in order: {numbers}.\n\n{request.prompt}"

    def complete(self, request: TurnRequest) -> str:
        with tempfile.TemporaryDirectory(prefix="advr-codex-") as scratch:
            out_file = os.path.join(scratch, "last_message.txt")
            started = time.monotonic()
            argv = self.build_argv(scratch, out_file, request.effort, request.images)
            proc = _run_cli(argv, self.codex_prompt(request), scratch, self.timeout)
            logger.info("codex role=%s round=%d effort=%s images=%d exit=%d %.1fs", request.role, request.round,
                        request.effort or "default", len(request.images), proc.returncode, time.monotonic() - started)
            if proc.returncode != 0:
                detail = codex_error_from_jsonl(proc.stdout) or _tail(proc.stderr)
                raise BackendError(f"codex exited {proc.returncode}: {detail}")
            if not os.path.exists(out_file):
                raise BackendError("codex produced no final message")
            with open(out_file, encoding="utf-8") as fh:
                text = fh.read().strip()
        if not text:
            raise BackendError("codex returned empty output")
        return text


def parse_gemini_json(stdout: str) -> Optional[Dict]:
    """The document printed by ``gemini -o json``, or None when stdout is not a JSON object."""
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def gemini_error_message(doc: Optional[Dict]) -> Optional[str]:
    """``{"error": {"type", "message", "code"}}`` as written by gemini on a failed headless run."""
    error = (doc or {}).get("error")
    if not error:
        return None
    return str(error.get("message") or error) if isinstance(error, dict) else str(error)


class GeminiBackend(Backend):
    """``gemini`` headless (Login with Google), read-only plan mode, no extensions, empty scratch dir."""

    name = "gemini"
    models = ("auto", "pro", "flash", "flash-lite")  # gemini -m aliases, CLI 0.66.0; no effort flag

    def __init__(self, model: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which("gemini") is not None

    def build_argv(self, model: Optional[str] = None) -> List[str]:
        # -p "" makes the run headless; the prompt itself comes on stdin (kept out of ps / argv limits).
        argv = ["gemini", *GEMINI_ISOLATION_FLAGS, "-o", "json"]
        model = model or self.model
        if model:
            argv += ["-m", model]
        return argv + ["-p", ""]

    def complete(self, request: TurnRequest) -> str:
        with tempfile.TemporaryDirectory(prefix="advr-gemini-") as scratch:
            started = time.monotonic()
            proc = _run_cli(self.build_argv(request.model), request.prompt, scratch, self.timeout)
            logger.info("gemini role=%s round=%d model=%s exit=%d %.1fs", request.role, request.round,
                        request.model or self.model or "default", proc.returncode, time.monotonic() - started)
        doc = parse_gemini_json(proc.stdout)
        error = gemini_error_message(doc)
        if proc.returncode != 0 or error:
            raise BackendError(f"gemini exited {proc.returncode}: {error or _tail(proc.stderr or proc.stdout)}")
        if doc is None:
            raise BackendError(f"gemini returned non-JSON output: {_tail(proc.stdout)}")
        text = doc.get("response")
        if not isinstance(text, str) or not text.strip():
            raise BackendError("gemini returned empty output")
        return text.strip()


class MockBackend(Backend):
    """Deterministic offline backend. The critic contests round 1 and accepts afterwards."""

    name = "mock"
    supports_images = True

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def available(self) -> bool:
        return True

    def complete(self, request: TurnRequest) -> str:
        if self.delay:
            time.sleep(self.delay)  # simulates model latency for the UI demo only
        return self._reply(request)

    def stream(self, request: TurnRequest, on_text: TextSink) -> str:
        reply = self._reply(request)
        words = re.findall(r"\S+\s*", reply)
        for word in words:
            if self.delay:
                time.sleep(self.delay / len(words))  # same total latency as complete(), spread over words
            on_text(word)
        return reply

    @staticmethod
    def _reply(request: TurnRequest) -> str:
        if request.role == "proposer" and request.round == 0:
            return (
                "**Position.** The claim holds under the stated assumptions.\n\n"
                "1. Premise A supports it.\n2. Premise B is consistent with known data.\n\n"
                "---PROPOSER_STATUS---\nCONFIDENCE: 70\nCONCEDED: 0\nPOSITION_CHANGED: false\n"
                "---END_PROPOSER_STATUS---"
            )
        if request.role == "proposer":
            return (
                f"**Rebuttal (round {request.round}).** I concede the edge case and narrow the claim.\n\n"
                "---PROPOSER_STATUS---\nCONFIDENCE: 80\nCONCEDED: 1\nPOSITION_CHANGED: true\n"
                "---END_PROPOSER_STATUS---"
            )
        if request.role == "chat":
            seen = f" and {len(request.images)} image(s)" if request.images else ""
            return f"**Mock reply #{request.round}.** I read {len(request.prompt)} characters of conversation{seen}."
        if request.role == "critic":
            verdict = "CONTEST" if request.round == 1 else "ACCEPT"
            objections = 2 if verdict == "CONTEST" else 0
            return (
                f"**Critique (round {request.round}).** "
                + ("Premise B ignores an edge case; `A` is asserted, not shown." if objections
                   else "The narrowed claim survives my objections.")
                + f"\n\n---CRITIC_STATUS---\nOBJECTIONS: {objections}\nVERDICT: {verdict}\n"
                "---END_CRITIC_STATUS---"
            )
        return (
            "## Verdict\nThe narrowed claim stands; the original overreached.\n\n"
            "---JUDGE_STATUS---\nCONFIDENCE: 75\nWINNER: draw\nUNRESOLVED: 0\n---END_JUDGE_STATUS---"
        )


DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"  # fallback when no key was saved from the UI
DEEPSEEK_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,128}")  # also keeps CR/LF out of the header
DEFAULT_KEYS_DIR = os.path.join(os.path.expanduser("~"), ".adversarial-review", "keys")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"  # OpenAI-compatible endpoint
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekBackend(Backend):
    """DeepSeek over its pay-per-use HTTP API (stdlib urllib). Enabled while its key store holds a key."""

    name = "deepseek"
    models = ("deepseek-chat", "deepseek-reasoner")

    def __init__(self, keys: ApiKeyStore, model: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, url: str = DEEPSEEK_URL) -> None:
        self.keys = keys
        self.model = model or DEEPSEEK_DEFAULT_MODEL
        self.timeout = timeout
        self.url = url

    def available(self) -> bool:
        return bool(self.keys.get())

    def complete(self, request: TurnRequest) -> str:
        api_key = self.keys.get()  # read per turn: a key saved or removed in the UI applies at once
        if not api_key:
            raise BackendError("deepseek: no API key (Comptes LLM → DeepSeek)")
        body = json.dumps({"model": request.model or self.model, "stream": False,
                           "messages": [{"role": "user", "content": request.prompt}]}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise BackendError(f"deepseek HTTP {exc.code}: {_deepseek_error(exc)}") from exc
        except socket.timeout as exc:
            raise BackendTimeout(f"deepseek did not answer within {self.timeout:.0f}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise BackendTimeout(f"deepseek did not answer within {self.timeout:.0f}s") from exc
            raise BackendError(f"deepseek unreachable: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise BackendError("deepseek returned non-JSON") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("deepseek response has no choices[0].message.content") from exc
        logger.info("deepseek %s round=%d answered in %.1fs", request.role, request.round, time.monotonic() - started)
        if not text or not text.strip():
            raise BackendError("deepseek returned an empty answer")
        return text.strip()


def _deepseek_error(exc: "urllib.error.HTTPError") -> str:
    """Typed API error message from the JSON body ({"error": {"message": ...}}), else the HTTP reason."""
    try:
        return json.loads(exc.read())["error"]["message"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return str(exc.reason)


def deepseek_key_store(keys_dir: str = DEFAULT_KEYS_DIR) -> ApiKeyStore:
    return ApiKeyStore(os.path.join(keys_dir, "deepseek"), DEEPSEEK_API_KEY_ENV, DEEPSEEK_KEY_PATTERN)


def build_registry(codex_model: Optional[str] = None, claude_model: Optional[str] = None,
                   timeout: float = DEFAULT_TIMEOUT_SECONDS, mock_delay: float = 0.0,
                   deepseek_model: Optional[str] = None, gemini_model: Optional[str] = None,
                   deepseek_keys: Optional[ApiKeyStore] = None) -> Dict[str, Backend]:
    return {
        "claude": ClaudeBackend(model=claude_model, timeout=timeout),
        "codex": CodexBackend(model=codex_model, timeout=timeout),
        "deepseek": DeepSeekBackend(deepseek_keys or deepseek_key_store(), model=deepseek_model, timeout=timeout),
        "gemini": GeminiBackend(model=gemini_model, timeout=timeout),
        "mock": MockBackend(delay=mock_delay),
    }
