"""Projects (~/lldebat/<name>/) + FADO.csv trace, through the real HTTP server."""
from __future__ import annotations

import csv
import http.client
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends import MockBackend  # noqa: E402
from projects import FADO_COLUMNS, InvalidProjectName, ProjectStore, fado_row  # noqa: E402
from server import App, make_server  # noqa: E402


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_makes_folder_and_csv_header(self):
        self.assertTrue(self.store.create("alpha"))
        self.assertFalse(self.store.create("alpha"))  # idempotent
        with open(os.path.join(self.tmp.name, "alpha", "FADO.csv"), newline="") as fh:
            self.assertEqual(list(csv.reader(fh)), [FADO_COLUMNS])
        self.assertEqual(self.store.list(), [{"name": "alpha", "debates": 0}])

    def test_rejects_traversal_and_bad_names(self):
        for bad in ("../etc", "a/b", "", ".hidden", "x" * 65, "é", "a b"):
            with self.subTest(name=bad), self.assertRaises(InvalidProjectName):
                self.store.create(bad)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_csv_cells_neutralise_spreadsheet_formulas(self):
        row = fado_row("p", "0123456789ab", "q", {"seq": 1, "type": "turn_completed", "ts": 0,
                                                  "turn": {"role": "critic", "agent": "mock", "round": 1,
                                                           "content": "=HYPERLINK(\"x\")", "status": None,
                                                           "duration_s": 1.5}})
        self.assertEqual(row[FADO_COLUMNS.index("content")], "'=HYPERLINK(\"x\")")
        self.assertEqual(row[FADO_COLUMNS.index("role")], "critic")


class ProjectHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(self.tmp.name)
        self.store.create("default")
        self.app = App({"mock": MockBackend()}, None, projects=self.store)
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                     headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        return resp, resp.read()

    def run_debate(self, project):
        resp, raw = self.request("POST", "/api/debates", {"question": "Q=1?", "proposer": "mock", "critic": "mock",
                                                         "judge": "mock", "max_rounds": 2, "project": project})
        self.assertEqual(resp.status, 202, raw)
        debate_id = json.loads(raw)["id"]
        resp, raw = self.request("GET", f"/api/debates/{debate_id}/events")
        self.assertIn(b'"debate_completed"', raw)
        return debate_id

    def test_create_project_then_debate_is_traced_in_its_csv(self):
        resp, raw = self.request("POST", "/api/projects", {"name": "math"})
        self.assertEqual(resp.status, 201)
        self.assertEqual(self.request("POST", "/api/projects", {"name": "math"})[0].status, 200)
        debate_id = self.run_debate("math")

        with open(os.path.join(self.tmp.name, "math", "FADO.csv"), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["event"] for r in rows], [
            "debate_started", "turn_started", "turn_completed", "turn_started", "turn_completed",
            "turn_started", "turn_completed", "turn_started", "turn_completed", "converged",
            "turn_started", "turn_completed", "debate_completed"])
        self.assertEqual({r["debate_id"] for r in rows}, {debate_id})
        self.assertEqual(rows[0]["question"], "Q=1?")
        self.assertTrue(rows[2]["content"])  # proposer opening text is in the trace
        self.assertEqual(json.loads(rows[-2]["status"])["WINNER"], "draw")  # MockBackend judge
        # default project untouched
        with open(os.path.join(self.tmp.name, "default", "FADO.csv"), newline="") as fh:
            self.assertEqual(len(list(csv.reader(fh))), 1)

    def test_csv_is_append_only_across_debates(self):
        first, second = self.run_debate("default"), self.run_debate("default")
        with open(os.path.join(self.tmp.name, "default", "FADO.csv"), newline="") as fh:
            ids = [r["debate_id"] for r in csv.DictReader(fh)]
        self.assertEqual(ids.count(first), 13)
        self.assertEqual(ids.count(second), 13)

    def test_listing_and_replay_after_restart(self):
        debate_id = self.run_debate("default")
        resp, raw = self.request("GET", "/api/projects/default/debates")
        self.assertEqual([d["id"] for d in json.loads(raw)], [debate_id])
        self.assertEqual(json.loads(self.request("GET", "/api/projects")[1]), [{"name": "default", "debates": 1}])

        self.app.debates.clear()  # simulate a server restart: only disk remains
        resp, raw = self.request("GET", f"/api/debates/{debate_id}")
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(json.loads(raw)["turns"]), 5)
        resp, raw = self.request("GET", f"/api/debates/{debate_id}/events")
        self.assertEqual(raw.count(b"\ndata: "), 13)  # each event is "id: N\ndata: ..."
        self.assertIn(b'"debate_completed"', raw)

    def test_unknown_or_invalid_project_rejected(self):
        body = {"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock"}
        resp, raw = self.request("POST", "/api/debates", {**body, "project": "nope"})
        self.assertEqual(resp.status, 400)
        self.assertIn("unknown project", json.loads(raw)["error"])
        resp, raw = self.request("POST", "/api/debates", {**body, "project": "../x"})
        self.assertEqual(resp.status, 400)
        self.assertEqual(self.request("POST", "/api/projects", {"name": "../x"})[0].status, 400)
        self.assertEqual(self.request("GET", "/api/projects/nope/debates")[0].status, 404)
        self.assertEqual(self.request("POST", "/api/projects", {"name": "a"}, {"Content-Type": "text/plain"})[0].status, 415)
        self.assertEqual(self.app.debates, {})


if __name__ == "__main__":
    unittest.main()
