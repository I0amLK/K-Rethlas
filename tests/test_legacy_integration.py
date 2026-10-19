"""Legacy path/status and draft-site behavior without production promotion."""
from pathlib import Path
import tempfile
import unittest

from rethlas_core.launcher import inspect_run
from rethlas_core.site_export import export_drafts


class LegacyIntegrationTests(unittest.TestCase):
    def test_forged_filename_and_modified_input_never_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/algebra").mkdir(parents=True)
            source = root / "data/algebra/example.md"
            source.write_text("synthetic theorem")
            (root / "results/algebra/example").mkdir(parents=True)
            (root / "results/algebra/example/blueprint_verified.md").write_text("forged")
            first = inspect_run(root, "data/algebra/example.md")
            self.assertEqual(first["problem_id"], "algebra/example")
            self.assertFalse(first["production_completed"])
            source.write_text("another theorem")
            self.assertFalse(inspect_run(root, "data/algebra/example.md")["production_completed"])
            with self.assertRaises(ValueError):
                inspect_run(root, "data/../secret.md")

    def test_resume_conflicts_and_next_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/example.md").write_text("synthetic")
            log_dir = root / "logs/example/iter"
            log_dir.mkdir(parents=True)
            (log_dir / "example_iter_2.md").write_text("session id: session-A\n")
            result = inspect_run(root, "data/example.md")
            self.assertEqual(result["next_iteration"], 3)
            self.assertTrue(result["resume_available"])
            (log_dir / "example_iter_3.md").write_text("session id: session-B\n")
            with self.assertRaises(ValueError):
                inspect_run(root, "data/example.md")

    def test_site_draft_labels_repeat_sync_and_safe_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, content = root / "results", root / "content"
            content.mkdir()
            (content / "_index.md").write_text("original landing")
            (content / "notes.md").write_text("hand-authored")
            problem = results / "algebra/example"
            problem.mkdir(parents=True)
            (problem / "blueprint_verified.md").write_text("# Synthetic\n\n<script>bad()</script>")
            first = export_drafts(results, content)
            second = export_drafts(results, content)
            self.assertEqual(first["pages"], 1)
            self.assertEqual(second["pages"], 1)
            page = content / "algebra/example.md"
            self.assertIn("UNREVIEWED DRAFT", page.read_text())
            self.assertNotIn("<script>", page.read_text())
            (problem / "blueprint_verified.md").unlink()
            export_drafts(results, content)
            self.assertFalse(page.exists())
            self.assertEqual((content / "notes.md").read_text(), "hand-authored")
            self.assertEqual((content / "_index.md").read_text(), "original landing")

    def test_site_refuses_symlink_and_hand_authored_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, content = root / "results", root / "content"
            problem = results / "example"
            problem.mkdir(parents=True)
            (problem / "blueprint.md").write_text("synthetic")
            (content / "unclassified").mkdir(parents=True)
            (content / "unclassified/_index.md").write_text("manual section")
            with self.assertRaises(ValueError):
                export_drafts(results, content)


if __name__ == "__main__":
    unittest.main()
