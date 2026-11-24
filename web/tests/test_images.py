"""Images in chats: validation, history labels and the per-reply cap, what the claude and codex
CLIs receive (fake executables on PATH, real pipes), and the HTTP routes (upload, serve, delete)."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import stat
import struct
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import zlib
from typing import List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attachments import MAX_IMAGE_BYTES, InvalidImage, parse_images, sniff_extension  # noqa: E402
from backends import (CLAUDE_STREAM_FLAGS, BackendError, ClaudeBackend, CodexBackend, ImageFile,  # noqa: E402
                      MockBackend, TurnRequest)
from chat import IMAGE_ONLY_TITLE, MAX_IMAGES_PER_REPLY, Chat, ChatConfig, ChatRunner, format_history  # noqa: E402
from engine import USER_ROLE, InvalidUserMessage, Turn  # noqa: E402
from server import MAX_BODY_BYTES, App, make_server  # noqa: E402
from test_claude_stream import FAKE_CLAUDE  # noqa: E402
from test_engine import ScriptedBackend  # noqa: E402


def png(rgb=(220, 20, 20), size=8, noise=b"") -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + (chunk(b"zzZz", noise) if noise else b"") + chunk(b"IEND", b""))


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
GIF = b"GIF89a" + b"\x00" * 20
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 10


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class ParseImagesTests(unittest.TestCase):
    def test_type_comes_from_the_bytes(self):
        images = parse_images([b64(png()), b64(JPEG), b64(GIF), b64(WEBP)])
        self.assertEqual([i.extension for i in images], ["png", "jpg", "gif", "webp"])
        self.assertEqual([i.media_type for i in images], ["image/png", "image/jpeg", "image/gif", "image/webp"])
        self.assertEqual(images[0].name, hashlib.sha256(png()).hexdigest() + ".png")
        self.assertEqual(images[0].data, png())

    def test_none_means_no_images(self):
        self.assertEqual(parse_images(None), [])

    def test_rejections(self):
        cases = [
            ("not a list", "images must be a list of base64 strings"),
            ([1], "images must be a list of base64 strings"),
            ([b64(JPEG)] * 5, "at most 4 images per message"),
            (["***"], "image 1 is not valid base64"),
            ([b64(JPEG), ""], "image 2 is empty"),
            ([b64(b"<svg onload=alert(1)>")], "image 1 is not a PNG, JPEG, GIF or WebP file"),
            ([b64(b"\xff\xd8\xff" + b"\x00" * MAX_IMAGE_BYTES)], "image 1 exceeds 5 MB"),
        ]
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaises(InvalidImage) as ctx:
                parse_images(raw)
            self.assertEqual(str(ctx.exception), message)
            self.assertIsInstance(ctx.exception, InvalidUserMessage)  # -> 400 like any bad message

    def test_riff_that_is_not_webp(self):
        self.assertIsNone(sniff_extension(b"RIFF\x10\x00\x00\x00WAVEfmt "))


def user(content: str, images=()) -> Turn:
    return Turn(role=USER_ROLE, agent=USER_ROLE, round=0, content=content, status=None, duration_s=0.0,
                images=list(images))


def assistant(content: str) -> Turn:
    return Turn(role="assistant", agent="m", round=0, content=content, status=None, duration_s=0.0)


class HistoryTests(unittest.TestCase):
    def test_images_are_numbered_across_the_chat(self):
        turns = [user("vois", ["a.png", "b.png"]), assistant("ok"), user("", ["c.jpg"])]
        self.assertEqual(format_history(turns),
                         "### USER\n[image 1]\n[image 2]\nvois\n\n### ASSISTANT\nok\n\n### USER\n[image 3]")

    def test_old_snapshot_turns_load_without_images(self):
        turn = Turn(**{"role": "user", "agent": "user", "round": 0, "content": "x", "status": None,
                       "duration_s": 0.0})
        self.assertEqual(turn.images, [])

    def test_photo_without_text_is_accepted_and_titled(self):
        chat = Chat(config=ChatConfig(agent="m"))
        chat.add_user_message("   ", ["a.png"])
        self.assertEqual((chat.turns[0].content, chat.title), ("", IMAGE_ONLY_TITLE))
        with self.assertRaisesRegex(InvalidUserMessage, "^message is empty$"):
            Chat(config=ChatConfig(agent="m")).add_user_message("   ")

    def test_only_the_newest_images_are_sent(self):
        chat = Chat(config=ChatConfig(agent="m"), files_dir="/files")
        chat.turns = [user("1", ["a.png", "b.png", "c.png"]), assistant("ok"), user("2", ["d.jpg", "e.gif"])]
        sent = ChatRunner.images_to_send(chat)
        self.assertEqual(MAX_IMAGES_PER_REPLY, 4)
        self.assertEqual(sent, (ImageFile(2, "/files/b.png", "image/png"), ImageFile(3, "/files/c.png", "image/png"),
                                ImageFile(4, "/files/d.jpg", "image/jpeg"), ImageFile(5, "/files/e.gif", "image/gif")))

    def test_images_of_omitted_messages_are_not_sent(self):
        chat = Chat(config=ChatConfig(agent="m"), files_dir="/f")
        chat.turns = [user("x" * 70_000, ["old.png"]), assistant("ok"), user("new", ["new.png"])]
        self.assertEqual(ChatRunner.images_to_send(chat), (ImageFile(2, "/f/new.png", "image/png"),))

    def test_images_without_a_folder_fail_the_reply_loudly(self):
        chat = Chat(config=ChatConfig(agent="s"))
        chat.add_user_message("vois", ["a.png"])
        ChatRunner({"s": ScriptedBackend(["never"])}).reply(chat)
        self.assertEqual(chat.events[-1]["type"], "reply_failed")
        self.assertEqual(chat.events[-1]["message"], "this chat has images but no folder to read them from")


class FakeCliCase(unittest.TestCase):
    script = ""
    binary = ""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, self.binary)
        with open(path, "w") as fh:
            fh.write(self.script)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        self.log = os.path.join(self.tmp.name, "call.json")
        self.env = mock.patch.dict(os.environ, {"PATH": self.tmp.name + os.pathsep + os.environ["PATH"],
                                                "FAKE_LOG": self.log, "FAKE_MODE": "ok"})
        self.env.start()
        self.red = os.path.join(self.tmp.name, "red.png")
        self.jpg = os.path.join(self.tmp.name, "pic.jpg")
        with open(self.red, "wb") as fh:
            fh.write(png())
        with open(self.jpg, "wb") as fh:
            fh.write(JPEG)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def call(self):
        with open(self.log) as fh:
            return json.load(fh)

    def request(self, images) -> TurnRequest:
        return TurnRequest(role="chat", round=1, prompt="### USER\n[image 3]\n[image 4]\nCouleurs ?", model="haiku",
                           images=tuple(images))


class ClaudeImageTests(FakeCliCase):
    script = FAKE_CLAUDE
    binary = "claude"

    def test_images_go_as_content_blocks_after_their_label(self):
        got: List[str] = []
        req = self.request([ImageFile(3, self.red, "image/png"), ImageFile(4, self.jpg, "image/jpeg")])
        self.assertEqual(ClaudeBackend().stream(req, got.append), "Bonjour")
        call = self.call()
        self.assertEqual(call["argv"], ["--print", "--setting-sources", "project", "--tools", "", "--strict-mcp-config",
                                        "--model", "haiku", *CLAUDE_STREAM_FLAGS, "--input-format", "stream-json"])
        self.assertTrue(call["prompt"].endswith("\n"))
        self.assertEqual(json.loads(call["prompt"]), {"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": "Image 3:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png())}},
            {"type": "text", "text": "Image 4:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(JPEG)}},
            {"type": "text", "text": "### USER\n[image 3]\n[image 4]\nCouleurs ?"}]}})

    def test_without_images_the_prompt_stays_plain_text(self):
        ClaudeBackend().stream(self.request([]), [].append)
        call = self.call()
        self.assertNotIn("--input-format", call["argv"])
        self.assertEqual(call["prompt"], "### USER\n[image 3]\n[image 4]\nCouleurs ?")

    def test_complete_refuses_images_instead_of_dropping_them(self):
        with self.assertRaisesRegex(BackendError, "^claude takes images only in stream mode$"):
            ClaudeBackend().complete(self.request([ImageFile(1, self.red, "image/png")]))
        self.assertFalse(os.path.exists(self.log), "the CLI must not run")


FAKE_CODEX = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, os, sys
    argv = sys.argv[1:]
    json.dump({"argv": argv, "prompt": sys.stdin.read()}, open(os.environ["FAKE_LOG"], "w"))
    with open(argv[argv.index("-o") + 1], "w") as fh:
        fh.write("Rouge et noir\\n")
    """)


class CodexImageTests(FakeCliCase):
    script = FAKE_CODEX
    binary = "codex"

    def test_each_image_is_one_i_flag_in_order_and_the_prompt_names_them(self):
        req = self.request([ImageFile(3, self.red, "image/png"), ImageFile(4, self.jpg, "image/jpeg")])
        self.assertEqual(CodexBackend(model="codex-auto-review").complete(req), "Rouge et noir")
        call = self.call()
        argv = call["argv"]
        scratch = argv[argv.index("-C") + 1]
        self.assertEqual(argv, ["exec", "--json", "-m", "codex-auto-review", "--sandbox", "read-only",
                                "--skip-git-repo-check", "-C", scratch, "-o", os.path.join(scratch, "last_message.txt"),
                                "-i", self.red, "-i", self.jpg, "-"])
        self.assertEqual(call["prompt"], "Attached images, in order: image 3, image 4.\n\n"
                                         "### USER\n[image 3]\n[image 4]\nCouleurs ?")

    def test_without_images_nothing_changes(self):
        CodexBackend().complete(self.request([]))
        call = self.call()
        self.assertNotIn("-i", call["argv"])
        self.assertEqual(call["prompt"], "### USER\n[image 3]\n[image 4]\nCouleurs ?")


class ImageRecorder(MockBackend):
    """Mock that keeps every request and checks the image files exist while the reply runs."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: List[TurnRequest] = []
        self.files_readable: List[bool] = []

    def stream(self, request, on_text):
        self.requests.append(request)
        self.files_readable.append(all(os.path.isfile(i.path) for i in request.images))
        return super().stream(request, on_text)


class ImageServerTests(unittest.TestCase):
    def setUp(self):
        from projects import ProjectStore
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(self.tmp.name)
        self.store.create("default")
        self.backend = ImageRecorder()
        self.app = App({"mock": self.backend, "blind": ScriptedBackend(["x"])}, None, max_active=1, projects=self.store)
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp, resp.read()

    def wait_idle(self, chat_id):
        deadline = time.monotonic() + 5
        while self.app.get_chat(chat_id).state == "busy" or not self.app.slots.acquire(blocking=False):
            self.assertLess(time.monotonic(), deadline, "reply never finished")
            threading.Event().wait(0.01)
        self.app.slots.release()

    def files_dir(self, chat_id):
        return self.store.chat_files_dir("default", chat_id)

    def test_upload_store_send_and_serve(self):
        big = png(noise=os.urandom(20_000))  # a private ancillary chunk: incompressible bulk
        self.assertGreater(len(b64(big)) * 2, MAX_BODY_BYTES, "must exceed the default body limit")
        resp, raw = self.request("POST", "/api/chats", {"agent": "mock", "text": "", "images": [b64(big), b64(big)]})
        self.assertEqual(resp.status, 202, raw)
        chat_id = json.loads(raw)["id"]
        self.wait_idle(chat_id)
        name = hashlib.sha256(big).hexdigest() + ".png"
        self.assertEqual(os.listdir(self.files_dir(chat_id)), [name])  # same photo twice: stored once
        snap = json.loads(self.request("GET", f"/api/chats/{chat_id}")[1])
        self.assertEqual(snap["turns"][0]["images"], [name, name])
        self.assertEqual(snap["title"], IMAGE_ONLY_TITLE)
        self.assertNotIn(b64(big)[:200], json.dumps(snap), "image bytes must not be in the chat JSON")
        self.assertEqual([i.number for i in self.backend.requests[0].images], [1, 2])
        self.assertEqual(self.backend.files_readable, [True])
        self.assertIn("2 image(s)", snap["turns"][1]["content"])

        resp, body = self.request("GET", f"/api/chats/{chat_id}/files/{name}")
        self.assertEqual((resp.status, body), (200, big))
        self.assertEqual(resp.getheader("Content-Type"), "image/png")
        self.assertEqual(resp.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.getheader("Content-Security-Policy"), "default-src 'none'; sandbox")

        resp, raw = self.request("POST", f"/api/chats/{chat_id}/messages", {"text": "et celle-ci ?",
                                                                            "images": [b64(JPEG)]})
        self.assertEqual(resp.status, 202, raw)
        self.wait_idle(chat_id)
        self.assertEqual([i.number for i in self.backend.requests[1].images], [1, 2, 3])

        self.app.chats.clear()  # restart: the reloaded chat must still find its images
        resp, body = self.request("GET", f"/api/chats/{chat_id}/files/{name}")
        self.assertEqual((resp.status, body), (200, big))

        resp, _ = self.request("DELETE", f"/api/chats/{chat_id}")
        self.assertEqual(resp.status, 204)
        self.assertFalse(os.path.exists(self.files_dir(chat_id)), "images must go with the chat")

    def test_bad_names_are_not_served(self):
        resp, raw = self.request("POST", "/api/chats", {"agent": "mock", "text": "x", "images": [b64(JPEG)]})
        chat_id = json.loads(raw)["id"]
        self.wait_idle(chat_id)
        good = hashlib.sha256(JPEG).hexdigest()
        for path in (f"{good}.svg", f"{good}.jpg.tmp", "..%2F..%2F" + good + ".jpg", "../" + chat_id + ".json",
                     ("0" * 64) + ".jpg"):
            with self.subTest(path=path):
                resp, _ = self.request("GET", f"/api/chats/{chat_id}/files/{path}")
                self.assertEqual(resp.status, 404)

    def test_rejected_images_store_nothing_and_cost_nothing(self):
        cases = [({"agent": "mock", "text": "x", "images": [b64(b"<svg/>")]}, "image 1 is not a PNG, JPEG, GIF or WebP file"),
                 ({"agent": "blind", "text": "x", "images": [b64(JPEG)]}, "blind cannot read images; pick claude or codex"),
                 ({"agent": "mock", "text": "", "images": []}, "message is empty")]
        for body, message in cases:
            with self.subTest(message=message):
                resp, raw = self.request("POST", "/api/chats", body)
                self.assertEqual((resp.status, json.loads(raw)["error"]), (400, message))
        self.assertEqual(self.backend.requests, [])
        self.assertEqual(os.listdir(os.path.join(self.tmp.name, "default", "chats")) if os.path.isdir(
            os.path.join(self.tmp.name, "default", "chats")) else [], [])

    def test_large_body_only_on_chat_routes(self):
        resp, raw = self.request("POST", "/api/debates", {"question": "x" * (MAX_BODY_BYTES + 1), "proposer": "mock",
                                                          "critic": "mock", "judge": "mock"})
        self.assertEqual((resp.status, json.loads(raw)["error"]), (400, f"body must be 1..{MAX_BODY_BYTES} bytes"))

    def test_index_csp_allows_blob_previews_only(self):
        resp, _ = self.request("GET", "/")
        self.assertIn("img-src 'self' blob:;", resp.getheader("Content-Security-Policy"))


if __name__ == "__main__":
    unittest.main()
