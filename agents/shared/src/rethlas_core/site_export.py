"""Local draft-only website export. Publication trust is never inferred from names."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .launcher import production_completed


MARKER = ".rethlas-generated.json"


def _regular_path(path: Path) -> bool:
    return not any(parent.is_symlink() for parent in (path, *path.parents))


def export_drafts(results: Path, content: Path) -> dict:
    results, content = results.absolute(), content.absolute()
    if not _regular_path(content) or not content.is_dir():
        raise ValueError("site content must be a real directory")
    if content == results or content in results.parents or results in content.parents:
        raise ValueError("site content and results must be separate")
    marker = content / MARKER
    if marker.is_symlink():
        raise ValueError("site export marker cannot be symlink")
    old = json.loads(marker.read_text()) if marker.exists() else []
    if not isinstance(old, list) or any(not isinstance(entry, str) for entry in old):
        raise ValueError("invalid site export history")
    for entry in old:
        path = Path(entry)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise ValueError("invalid site export history")
    generated = []
    if results.exists():
        if not _regular_path(results) or not results.is_dir():
            raise ValueError("results must be a real directory")
        directories = sorted({path.parent for name in ("blueprint.md", "blueprint_verified.md")
                              for path in results.rglob(name) if _regular_path(path)})
        for directory in directories:
            relative = directory.relative_to(results)
            if not relative.parts or any(not re.fullmatch(r"[\w.-]+", part) or part in {".", ".."} for part in relative.parts):
                raise ValueError("unsafe problem path")
            source = directory / "blueprint.md"
            if not source.is_file():
                source = directory / "blueprint_verified.md"
            if not _regular_path(source) or not source.is_file() or source.stat().st_size > 1048576:
                raise ValueError("invalid draft body")
            # Read exactly once. Raw HTML is escaped for legacy untrusted drafts;
            # a site page is not a proof-review artifact.
            body = source.read_text(encoding="utf-8").replace("<", "&lt;").replace(">", "&gt;")
            category = Path(*relative.parts[:-1]) if len(relative.parts) > 1 else Path("unclassified")
            for parent in (category, *category.parents):
                if parent == Path("."):
                    continue
                target_dir = content / parent
                if not _regular_path(target_dir):
                    raise ValueError("site path contains symlink")
                target_dir.mkdir(parents=True, exist_ok=True)
                section = target_dir / "_index.md"
                if section.is_symlink():
                    raise ValueError("section path contains symlink")
                section_name = section.relative_to(content).as_posix()
                if section.exists() and section_name not in old:
                    raise ValueError("refusing to replace hand-authored section")
                if not section.exists():
                    section.write_text("+++\ntitle = " + json.dumps(parent.name, ensure_ascii=False) + "\nsort_by = \"slug\"\n+++\n")
                if section_name not in generated:
                    generated.append(section_name)
            destination = content / category / (relative.name + ".md")
            if destination.is_symlink() or (destination.exists() and destination.relative_to(content).as_posix() not in old):
                raise ValueError("refusing to overwrite non-generated website page")
            assert production_completed() is False
            title = json.dumps(relative.name, ensure_ascii=False)
            destination.write_text(f"+++\ntitle = {title}\n[extra]\nmath = true\nreview_status = \"unreviewed\"\n+++\n\n"
                "> **UNREVIEWED DRAFT — no trusted mathematical review receipt.**\n\n" + body, encoding="utf-8")
            generated.append(destination.relative_to(content).as_posix())
    # Only remove files from our previous explicit export list, never arbitrary
    # directories or hand-authored content. Inspect each target before deletion.
    for entry in old:
        path = Path(entry)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise ValueError("invalid site export history")
        target = content / path
        if entry not in generated and _regular_path(target) and target.is_file():
            target.unlink()
    marker.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"pages": sum(Path(entry).name != "_index.md" for entry in generated),
            "review_status": "unreviewed", "production_completed": False}
