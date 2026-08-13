from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "TUTORIAL.md"
DOCS = ROOT / "docs"
PDF_NAME = "lora-finetune-studio-zero-to-mastery.pdf"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.has_main = False
        self.has_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        if tag == "a" and (href := attributes.get("href")):
            self.hrefs.append(href)
        if tag == "main":
            self.has_main = True
        if tag == "title":
            self.has_title = True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tutorial_has_complete_progression() -> None:
    content = SOURCE.read_text(encoding="utf-8")
    headings = re.findall(r"(?m)^## (.+)$", content)

    assert len(headings) == 22
    assert headings[0].startswith("Module 1")
    assert any(heading.startswith("Module 12 - Lab A") for heading in headings)
    assert any(heading.startswith("Module 13 - Lab B") for heading in headings)
    assert any(heading.startswith("Module 17 - Capstone") for heading in headings)
    assert headings[-1].startswith("Appendix E")
    for expected in (
        "Supervised Fine-Tuning",
        "Reward Modeling",
        "DPO",
        "KTO",
        "ORPO",
        "LoRA",
        "QLoRA",
        "OFT",
        "QOFT",
        "Unsloth",
    ):
        assert expected in content


def test_preference_lab_dataset_is_well_formed() -> None:
    path = ROOT / "examples" / "preference_sample.jsonl"
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 20
    for record in records:
        assert set(record) == {"prompt", "chosen", "rejected"}
        assert all(
            isinstance(value, str) and value.strip() for value in record.values()
        )
        assert record["chosen"] != record["rejected"]


def test_generated_manifest_matches_files() -> None:
    manifest = json.loads((DOCS / ".tutorial-build.json").read_text(encoding="utf-8"))

    assert manifest["source"] == "TUTORIAL.md"
    assert "index.html" in manifest["files"]
    assert f"downloads/{PDF_NAME}" in manifest["files"]
    assert len([path for path in manifest["files"] if path.endswith(".html")]) == 23
    for relative, expected_digest in manifest["files"].items():
        path = DOCS / relative
        assert path.is_file(), relative
        assert sha256(path) == expected_digest, relative


def test_generated_pages_have_valid_local_links() -> None:
    pages = sorted(DOCS.glob("*.html"))
    assert len(pages) == 23

    for page in pages:
        parser = PageParser()
        parser.feed(page.read_text(encoding="utf-8"))
        assert parser.has_main, page.name
        assert parser.has_title, page.name
        for href in parser.hrefs:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            target_path = unquote(parsed.path)
            target = page if not target_path else (page.parent / target_path).resolve()
            assert target.exists(), f"{page.name}: {href}"
            if parsed.fragment and target == page.resolve():
                assert parsed.fragment in parser.ids, f"{page.name}: {href}"


def test_published_and_canonical_pdfs_are_identical() -> None:
    canonical = ROOT / "output" / "pdf" / PDF_NAME
    published = DOCS / "downloads" / PDF_NAME

    assert canonical.read_bytes().startswith(b"%PDF-")
    assert canonical.stat().st_size > 100_000
    assert canonical.read_bytes() == published.read_bytes()
