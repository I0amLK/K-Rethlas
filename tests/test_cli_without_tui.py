"""Existing CLI entries remain usable without optional terminal UI imports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

from rethlas_core.private_fs import canonical_json
from test_project_service import ProjectFixtureMixin


ROOT = Path(__file__).resolve().parents[1]
BLOCK_UI = r'''
import importlib.abc
import runpy
import sys

class NoUI(importlib.abc.MetaPathFinder):
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"textual", "mdit_py_plugins", "linkify_it", "uc_micro"}:
            self.attempts.append(fullname)
            raise ModuleNotFoundError("optional UI import blocked: " + fullname)

blocker = NoUI()
sys.meta_path.insert(0, blocker)
try:
    import textual
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("Textual import guard is ineffective")
assert blocker.attempts == ["textual"]
blocker.attempts.clear()
kind, target = sys.argv[1:3]
sys.argv = [target, *sys.argv[3:]]
code = 0
try:
    if kind == "wrapper":
        runpy.run_path(target, run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__")
except SystemExit as exc:
    code = exc.code
assert blocker.attempts == [], blocker.attempts
assert not any(name == "textual" or name.startswith("textual.") for name in sys.modules)
raise SystemExit(code)
'''


class RetiredTUIFilesTests(unittest.TestCase):
    def test_ui_modules_wrappers_and_test_hosts_are_absent(self):
        for name in ("tui", "tui_controller"):
            with self.subTest(module=name):
                self.assertIsNone(importlib.util.find_spec("rethlas_core." + name))
                self.assertFalse((ROOT / "agents/shared/src/rethlas_core" / (name + ".py")).exists())
        for name in ("tools/research_tui.py", "tools/verify_codex_local.py", "tests/test_tui.py",
                     "tests/test_tui_controller.py", "tests/test_codex_operator_check.py",
                     "tests/tui_support.py", "tests/tui_pty_app.py", "tests/drive_tui_pty.py",
                     "tests/drive_tui_controls_pty.py"):
            with self.subTest(path=name):
                self.assertFalse((ROOT / name).exists())
        self.assertTrue((ROOT / "tests/test_tui_backend.py").is_file())

    def test_pdf_extra_remains_without_ui_dependencies(self):
        project = tomllib.loads((ROOT / "agents/shared/pyproject.toml").read_text())["project"]
        extras = project["optional-dependencies"]
        self.assertNotIn("tui", extras)
        self.assertEqual(extras["sources"], ["pypdf==6.19.0"])
        dependencies = project["dependencies"] + [value for group in extras.values() for value in group]
        for value in dependencies:
            self.assertNotIn(value.split("==")[0],
                             {"textual", "mdit-py-plugins", "linkify-it-py", "uc-micro-py"})


class CLIWithoutTUITests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.home = self.base / "empty-home"
        self.home.mkdir(mode=0o700)
        credentials = self.base / "credentials"
        credentials.mkdir(mode=0o700)
        token_path = credentials / "reader.token"
        token_path.write_text(self.reader["token"], encoding="ascii")
        token_path.chmod(0o600)
        self.config_path = self.base / "client.json"
        self.config_path.write_bytes(canonical_json({"version": "research-client-3", "root": str(self.service.root),
            "reader": {"run_id": self.reader["run_id"], "token_file": str(token_path)},
            "writer": None, "indexer": None, "runner": None, "embedding": {"mode": "lexical"},
            "generation_provider": {"type": "codex_cli"}, "review_provider": {"type": "codex_cli"}}))
        self.config_path.chmod(0o600)

    def invoke(self, kind, target, *arguments):
        process = subprocess.run([sys.executable, "-I", "-B", "-c", BLOCK_UI, kind, str(target), *map(str, arguments)],
            cwd=self.home, stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
            env={"PATH": "/nonexistent", "LANG": "C.UTF-8", "HOME": str(self.home),
                 "CODEX_HOME": str(self.home / "unused-codex")})
        self.assertEqual(process.returncode, 0, process.stdout.decode() + process.stderr.decode())
        self.assertEqual(process.stderr, b"")
        self.assertNotIn(self.reader["token"].encode(), process.stdout)
        return process.stdout

    def test_existing_wrapper_and_module_help_without_textual(self):
        for kind, target in (("wrapper", ROOT / "tools/research.py"), ("module", "rethlas_core.research_cli")):
            with self.subTest(entry=kind):
                output = self.invoke(kind, target, "--help")
                self.assertIn(b"rethlas-research", output)
                self.assertIn(b"research-history", output)

    def test_existing_wrapper_and_module_status_without_textual(self):
        for kind, target in (("wrapper", ROOT / "tools/research.py"), ("module", "rethlas_core.research_cli")):
            with self.subTest(entry=kind):
                value = json.loads(self.invoke(kind, target, "status", "--config", self.config_path))
                # Opening ProjectService may reconcile its local control journal.
                self.assertEqual(value, self.call("status", self.reader))
                self.assertIn("claim:0", value["current"])
        self.assertEqual(list(self.home.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
