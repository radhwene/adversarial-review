"""ApiKeyStore: file mode, validation without echo, env fallback precedence."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from api_keys import InvalidApiKey  # noqa: E402
from backends import deepseek_key_store  # noqa: E402

KEY = "sk-0123456789abcdefABCDEFwxyz"
ENV_KEY = "sk-fromenvironment0000000000"


class ApiKeyStoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        self.root = tmp.name
        self.store = deepseek_key_store(self.root)
        self.path = os.path.join(self.root, "deepseek", "api_key")

    def test_empty(self):
        self.assertEqual(self.store.get(), "")
        self.assertEqual(self.store.status(), {"configured": False, "source": None, "hint": None})

    def test_set_strips_paste_whitespace_and_is_private(self):
        status = self.store.set(f"  {KEY}\r\n")
        self.assertEqual(status, {"configured": True, "source": "file", "hint": "…wxyz"})
        self.assertEqual(self.store.get(), KEY)
        with open(self.path) as fh:
            self.assertEqual(fh.read(), KEY)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(self.path)).st_mode), 0o700)
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["api_key"])  # no temp file left
        self.assertNotIn(KEY, repr(status))

    def test_invalid_keys_rejected_without_echo(self):
        bad = ["", "sk-short", "pk-0123456789abcdefABCDEFwxyz", KEY + "\r\nX-Evil: 1", "sk-" + "a" * 129,
               "sk-0123456789abcdef ABCDEFwxyz"]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(InvalidApiKey) as ctx:
                self.store.set(value)
            if value:
                self.assertNotIn(value.strip(), str(ctx.exception))
        self.assertFalse(os.path.exists(self.path))

    def test_overwrite_and_clear(self):
        self.store.set(KEY)
        self.store.set("sk-" + "b" * 24)
        self.assertEqual(self.store.get(), "sk-" + "b" * 24)
        self.assertEqual(self.store.clear(), {"configured": False, "source": None, "hint": None})
        self.assertFalse(os.path.exists(self.path))
        self.store.clear()  # idempotent

    def test_file_wins_over_env_and_env_survives_clear(self):
        os.environ["DEEPSEEK_API_KEY"] = ENV_KEY
        self.assertEqual((self.store.get(), self.store.source()), (ENV_KEY, "env"))
        self.store.set(KEY)
        self.assertEqual((self.store.get(), self.store.source()), (KEY, "file"))
        self.assertEqual(self.store.clear(), {"configured": True, "source": "env", "hint": "…0000"})


if __name__ == "__main__":
    unittest.main()
