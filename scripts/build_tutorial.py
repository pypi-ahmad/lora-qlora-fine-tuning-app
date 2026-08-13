"""Build the tutorial website and PDF from TUTORIAL.md.

Usage:
    uv run --group docs python scripts/build_tutorial.py
    uv run --group docs python scripts/build_tutorial.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from markdown import markdown
from reportlab import rl_config

rl_config.invariant = 1

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TUTORIAL.md"
ASSET_SOURCE = ROOT / "scripts" / "tutorial_assets"
DOCS_ROOT = ROOT / "docs"
PDF_ROOT = ROOT / "output" / "pdf"
PDF_NAME = "lora-finetune-studio-zero-to-mastery.pdf"
MANIFEST_NAME = ".tutorial-build.json"
TITLE = "LoRA Fine-tune Studio: Zero to Mastery"
SUBTITLE = (
    "NLP, transformers, parameter-efficient fine-tuning, preference optimization, "
    "evaluation, and this repository"
)

MARKDOWN_EXTENSIONS = [
    "fenced_code",
    "tables",
    "sane_lists",
    "toc",
]


@dataclass(frozen=True, slots=True)
class Chapter:
    number: int
    title: str
    slug: str
    markdown: str
    html: str
    summary: str

    @property
    def filename(self) -> str:
        return f"{self.number:02d}-{self.slug}.html"


def slugify(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "chapter"


def split_source(source: str) -> tuple[str, list[tuple[str, str]]]:
    headings = list(re.finditer(r"(?m)^## (.+?)\s*$", source))
    if not headings:
        raise ValueError("TUTORIAL.md must contain at least one level-two chapter.")
    preamble = source[: headings[0].start()].strip()
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(source)
        title = match.group(1).strip()
        body = source[match.end() : end].strip()
        sections.append((title, body))
    return preamble, sections


def render_markdown(value: str) -> str:
    return markdown(
        value,
        extensions=MARKDOWN_EXTENSIONS,
        extension_configs={"toc": {"permalink": False}},
        output_format="html5",
    )


def plain_text(value: str) -> str:
    return " ".join(BeautifulSoup(value, "html.parser").get_text(" ").split())


def pdf_safe(value: str) -> str:
    """Replace glyphs absent from ReportLab's built-in fonts."""
    replacements = {
        "→": "->",
        "←": "<-",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "×": "x",
        "≈": "~=",
        "≤": "<=",
        "≥": ">=",
    }
    return "".join(replacements.get(character, character) for character in value)


def chapter_summary(rendered: str) -> str:
    soup = BeautifulSoup(rendered, "html.parser")
    for paragraph in soup.find_all("p"):
        text = " ".join(paragraph.get_text(" ").split())
        if len(text) >= 40:
            return text[:237].rstrip() + ("..." if len(text) > 237 else "")
    return "A chapter in the Zero-to-Mastery course."


def load_course() -> tuple[str, str, list[Chapter]]:
    source = SOURCE.read_text(encoding="utf-8")
    preamble_markdown, raw_sections = split_source(source)
    preamble_html = render_markdown(preamble_markdown)
    chapters: list[Chapter] = []
    used_slugs: set[str] = set()
    for number, (title, body) in enumerate(raw_sections, start=1):
        slug = slugify(
            re.sub(
                r"^(module|appendix)\s+[a-z0-9]+\s*[-:]\s*",
                "",
                title,
                flags=re.IGNORECASE,
            )
        )
        if slug in used_slugs:
            slug = f"{slug}-{number}"
        used_slugs.add(slug)
        rendered = render_markdown(f"## {title}\n\n{body}")
        chapters.append(
            Chapter(
                number=number,
                title=title,
                slug=slug,
                markdown=body,
                html=rendered,
                summary=chapter_summary(rendered),
            )
        )
    return source, preamble_html, chapters


def nav_items(chapters: list[Chapter], current: str | None = None) -> str:
    parts: list[str] = []
    for chapter in chapters:
        active = (
            ' class="active" aria-current="page"' if chapter.filename == current else ""
        )
        label = html.escape(chapter.title)
        parts.append(
            f'<li><a href="{chapter.filename}"{active}>'
            f"<span>{chapter.number:02d}</span>{label}</a></li>"
        )
    return "\n".join(parts)


def site_shell(
    *,
    title: str,
    description: str,
    content: str,
    chapters: list[Chapter],
    current: str | None,
    page_class: str,
) -> str:
    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    nav = nav_items(chapters, current)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{escaped_description}">
  <meta name="theme-color" content="#171a26">
  <title>{escaped_title}</title>
  <link rel="stylesheet" href="assets/site.css">
  <script defer src="assets/search-index.js"></script>
  <script defer src="assets/site.js"></script>
</head>
<body class="{page_class}" data-page="{html.escape(current or "index.html")}">
  <a class="skip-link" href="#main-content">Skip to content</a>
  <div class="reading-progress" aria-hidden="true"><span id="reading-progress"></span></div>
  <header class="topbar">
    <a class="brand" href="index.html" aria-label="Zero to Mastery home">
      <span class="brand-mark" aria-hidden="true">A<sub>r</sub></span>
      <span><strong>LoRA Studio</strong><small>Zero to Mastery</small></span>
    </a>
    <div class="topbar-actions">
      <button class="search-button" type="button" data-search-open aria-haspopup="dialog">
        <span aria-hidden="true">⌕</span> Search <kbd>Ctrl K</kbd>
      </button>
      <a class="pdf-link" href="downloads/{PDF_NAME}">Download PDF</a>
      <button class="nav-button" type="button" data-nav-toggle aria-expanded="false" aria-controls="course-nav">Contents</button>
    </div>
  </header>
  <div class="site-grid">
    <aside class="course-nav" id="course-nav">
      <div class="course-meter"><span>Course progress</span><strong data-course-progress>0%</strong><div><i data-course-progress-bar></i></div></div>
      <nav aria-label="Course chapters"><ol>{nav}</ol></nav>
      <div class="nav-meta"><a href="https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/blob/main/TUTORIAL.md">Markdown source</a><a href="https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/blob/main/TECHNICAL.md">Technical reference</a></div>
    </aside>
    <main id="main-content" tabindex="-1">{content}</main>
  </div>
  <footer class="site-footer"><p>Generated from <code>TUTORIAL.md</code>. Repository code and passing tests define app behavior.</p></footer>
  <dialog class="search-dialog" data-search-dialog aria-labelledby="search-title">
    <form method="dialog"><button aria-label="Close search">Close</button></form>
    <h2 id="search-title">Search the handbook</h2>
    <input type="search" data-search-input placeholder="Try attention, QLoRA, beta, checkpoint..." autocomplete="off">
    <p class="search-hint">Search chapter titles, headings, and course text.</p>
    <ol data-search-results></ol>
  </dialog>
</body>
</html>
"""


def index_content(preamble_html: str, chapters: list[Chapter]) -> str:
    cards = "\n".join(
        f"""<article class="chapter-card">
  <a href="{chapter.filename}">
    <span class="chapter-index">{chapter.number:02d}</span>
    <h2>{html.escape(chapter.title)}</h2>
    <p>{html.escape(chapter.summary)}</p>
    <strong>Open chapter <span aria-hidden="true">→</span></strong>
  </a>
</article>"""
        for chapter in chapters
    )
    return f"""<section class="hero">
  <div class="hero-copy">
    <p class="eyebrow">An applied local-LLM curriculum</p>
    <h1>From first token<br>to defensible adapter.</h1>
    <p>{html.escape(SUBTITLE)}.</p>
    <div class="hero-actions"><a class="primary-action" href="{chapters[0].filename}">Start the course</a><a href="downloads/{PDF_NAME}">Read offline</a></div>
  </div>
  <div class="token-ribbon" aria-label="Course progression from text to adapter">
    <span>text</span><i>→</i><span>tokens</span><i>→</i><span>vectors</span><i>→</i><span>attention</span><i>→</i><span>loss</span><i>→</i><span>adapter</span>
  </div>
</section>
<section class="course-intro prose">{preamble_html}</section>
<section class="course-map" aria-labelledby="course-map-title">
  <div class="section-heading"><p class="eyebrow">Course map</p><h2 id="course-map-title">Learn, run, measure, and maintain.</h2></div>
  <div class="chapter-grid">{cards}</div>
</section>"""


def chapter_content(chapter: Chapter, chapters: list[Chapter]) -> str:
    previous = chapters[chapter.number - 2] if chapter.number > 1 else None
    next_chapter = chapters[chapter.number] if chapter.number < len(chapters) else None
    previous_link = (
        f'<a rel="prev" href="{previous.filename}"><span>Previous</span><strong>{html.escape(previous.title)}</strong></a>'
        if previous
        else '<a rel="prev" href="index.html"><span>Previous</span><strong>Course home</strong></a>'
    )
    next_link = (
        f'<a rel="next" href="{next_chapter.filename}"><span>Next</span><strong>{html.escape(next_chapter.title)}</strong></a>'
        if next_chapter
        else '<a rel="next" href="index.html"><span>Complete</span><strong>Return to course map</strong></a>'
    )
    rendered = rewrite_root_links(chapter.html)
    return f"""<article class="chapter">
  <header class="chapter-hero"><p class="eyebrow">Chapter {chapter.number:02d} of {len(chapters):02d}</p><div class="chapter-signal" aria-hidden="true"><span style="--chapter:{chapter.number};--total:{len(chapters)}"></span></div></header>
  <div class="prose">{rendered}</div>
  <nav class="chapter-pagination" aria-label="Adjacent chapters">{previous_link}{next_link}</nav>
</article>"""


def rewrite_root_links(rendered: str) -> str:
    """Make root-relative Markdown links work from generated docs pages."""
    soup = BeautifulSoup(rendered, "html.parser")
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if href.startswith(("#", "http://", "https://", "mailto:")):
            continue
        path, separator, fragment = href.partition("#")
        if path and (ROOT / path).exists():
            link["href"] = (
                "https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/"
                f"blob/main/{path}{separator}{fragment}"
            )
    return str(soup)


def build_search_index(chapters: list[Chapter]) -> str:
    records: list[dict[str, str]] = []
    for chapter in chapters:
        soup = BeautifulSoup(chapter.html, "html.parser")
        headings = [
            heading.get_text(" ", strip=True)
            for heading in soup.find_all(["h2", "h3", "h4"])
        ]
        records.append(
            {
                "title": chapter.title,
                "url": chapter.filename,
                "summary": chapter.summary,
                "headings": " | ".join(headings),
                "text": " ".join(soup.get_text(" ").split()),
            }
        )
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return f"window.HANDBOOK_SEARCH={payload};\n"


def build_site_files(
    preamble_html: str, chapters: list[Chapter], pdf_bytes: bytes
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    files[".nojekyll"] = b""
    index = site_shell(
        title=TITLE,
        description=SUBTITLE,
        content=index_content(preamble_html, chapters),
        chapters=chapters,
        current=None,
        page_class="home-page",
    )
    files["index.html"] = index.encode("utf-8")
    for chapter in chapters:
        page = site_shell(
            title=f"{chapter.title} | {TITLE}",
            description=chapter.summary,
            content=chapter_content(chapter, chapters),
            chapters=chapters,
            current=chapter.filename,
            page_class="chapter-page",
        )
        files[chapter.filename] = page.encode("utf-8")
    files["assets/site.css"] = (ASSET_SOURCE / "site.css").read_bytes()
    files["assets/site.js"] = (ASSET_SOURCE / "site.js").read_bytes()
    files["assets/search-index.js"] = build_search_index(chapters).encode("utf-8")
    files[f"downloads/{PDF_NAME}"] = pdf_bytes
    return files


class HandbookDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str | Path, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        self.current_heading = TITLE
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[frame], onPage=self._cover_page),
                PageTemplate(id="body", frames=[frame], onPage=self._body_page),
            ]
        )

    def _metadata(self, canvas: Any) -> None:
        canvas.setTitle(TITLE)
        canvas.setAuthor("LoRA Fine-tune Studio")
        canvas.setSubject(SUBTITLE)
        canvas.setKeywords("NLP, transformers, LoRA, QLoRA, PEFT, fine-tuning")

    def _cover_page(self, canvas: Any, doc: Any) -> None:
        del doc
        self._metadata(canvas)

    def _body_page(self, canvas: Any, doc: Any) -> None:
        self._metadata(canvas)
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DDEA"))
        canvas.setLineWidth(0.5)
        canvas.line(
            doc.leftMargin, A4[1] - 15 * mm, A4[0] - doc.rightMargin, A4[1] - 15 * mm
        )
        canvas.setFillColor(colors.HexColor("#5D6475"))
        canvas.setFont("Helvetica", 7.5)
        heading = TITLE
        max_width = doc.width - 25 * mm
        while stringWidth(heading, "Helvetica", 7.5) > max_width and len(heading) > 12:
            heading = heading[:-2].rstrip() + "..."
        canvas.drawString(doc.leftMargin, A4[1] - 11.5 * mm, heading)
        canvas.drawRightString(
            A4[0] - doc.rightMargin, 10 * mm, str(canvas.getPageNumber())
        )
        canvas.restoreState()

    def afterFlowable(self, flowable: Any) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"PDFChapter", "PDFSection"}:
            return
        level = 0 if style_name == "PDFChapter" else 1
        text = flowable.getPlainText()
        key = getattr(flowable, "_bookmarkName", None)
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))
        if level == 0:
            self.current_heading = text


def pdf_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    ink = colors.HexColor("#1D2230")
    muted = colors.HexColor("#5D6475")
    violet = colors.HexColor("#6952D5")
    return {
        "body": ParagraphStyle(
            "PDFBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.4,
            textColor=ink,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "small": ParagraphStyle(
            "PDFSmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10.2,
            textColor=ink,
        ),
        "chapter": ParagraphStyle(
            "PDFChapter",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#262B3B"),
            spaceBefore=3,
            spaceAfter=12,
            keepWithNext=True,
        ),
        "section": ParagraphStyle(
            "PDFSection",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=violet,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "PDFSubsection",
            parent=sample["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#167981"),
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "PDFCode",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=7.1,
            leading=9.2,
            textColor=colors.HexColor("#E9ECF5"),
            backColor=colors.HexColor("#242938"),
            borderPadding=7,
            borderRadius=3,
            spaceBefore=4,
            spaceAfter=8,
            splitLongWords=True,
        ),
        "caption": ParagraphStyle(
            "PDFCaption",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.5,
            textColor=muted,
        ),
        "toc": ParagraphStyle(
            "PDFTOC",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=ink,
            leftIndent=14,
            firstLineIndent=-14,
        ),
    }


def inline_markup(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return html.escape(pdf_safe(str(node)))
    inner = "".join(inline_markup(child) for child in node.children)
    if node.name in {"strong", "b"}:
        return f"<b>{inner}</b>"
    if node.name in {"em", "i"}:
        return f"<i>{inner}</i>"
    if node.name == "code":
        return f'<font name="Courier" color="#503AA8">{inner}</font>'
    if node.name == "a":
        href = html.escape(str(node.get("href", "")), quote=True)
        return f'<a href="{href}" color="#4D46B8">{inner}</a>'
    if node.name == "br":
        return "<br/>"
    return inner


def paragraph_from_tag(tag: Tag, style: ParagraphStyle) -> Paragraph:
    content = "".join(inline_markup(child) for child in tag.children).strip()
    return Paragraph(content or "&#160;", style)


def table_flowable(
    table_tag: Tag, styles: dict[str, ParagraphStyle], width: float
) -> Table:
    rows: list[list[Paragraph]] = []
    for row_index, row in enumerate(table_tag.find_all("tr")):
        cells = row.find_all(["th", "td"], recursive=False)
        row_values: list[Paragraph] = []
        for cell in cells:
            markup = "".join(inline_markup(child) for child in cell.children).strip()
            if row_index == 0 or cell.name == "th":
                markup = f"<b>{markup}</b>"
            row_values.append(Paragraph(markup or "&#160;", styles["small"]))
        if row_values:
            rows.append(row_values)
    if not rows:
        return Table([[""]], colWidths=[width])
    columns = max(len(row) for row in rows)
    for row in rows:
        row.extend(Paragraph("", styles["small"]) for _ in range(columns - len(row)))
    if columns == 2:
        col_widths = [width * 0.31, width * 0.69]
    else:
        col_widths = [width / columns] * columns
    result = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEBFA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#27243A")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8F9FC")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CDD3E0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def list_flowable(tag: Tag, styles: dict[str, ParagraphStyle]) -> ListFlowable:
    ordered = tag.name == "ol"
    items: list[ListItem] = []
    for item in tag.find_all("li", recursive=False):
        blocks: list[Any] = []
        direct_text: list[str] = []
        for child in item.children:
            if isinstance(child, NavigableString):
                direct_text.append(html.escape(str(child)))
            elif child.name in {"ul", "ol"}:
                if "".join(direct_text).strip():
                    blocks.append(Paragraph("".join(direct_text), styles["body"]))
                    direct_text = []
                blocks.append(list_flowable(child, styles))
            else:
                direct_text.append(inline_markup(child))
        if "".join(direct_text).strip():
            blocks.insert(0, Paragraph("".join(direct_text), styles["body"]))
        items.append(ListItem(blocks or [Paragraph("", styles["body"])], leftIndent=12))
    options: dict[str, Any] = {
        "bulletType": "1" if ordered else "bullet",
        "leftIndent": 18,
        "bulletFontName": "Helvetica",
        "bulletFontSize": 8,
        "spaceAfter": 5,
    }
    if ordered:
        options["start"] = "1"
    else:
        options["start"] = "-"
    return ListFlowable(items, **options)


def html_to_flowables(
    rendered: str,
    styles: dict[str, ParagraphStyle],
    width: float,
    bookmark_prefix: str,
) -> list[Any]:
    soup = BeautifulSoup(rendered, "html.parser")
    flowables: list[Any] = []
    heading_counter = 0
    for node in soup.contents:
        if isinstance(node, NavigableString) or not isinstance(node, Tag):
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            heading_counter += 1
            level_style = {
                "h1": styles["chapter"],
                "h2": styles["chapter"],
                "h3": styles["section"],
                "h4": styles["subsection"],
            }[node.name]
            heading_text = pdf_safe(node.get_text(" ", strip=True))
            bookmark = (
                f"{bookmark_prefix}-heading-{heading_counter}-{slugify(heading_text)}"
            )
            paragraph = Paragraph(
                f'<a name="{bookmark}"/>{html.escape(heading_text)}',
                level_style,
            )
            paragraph._bookmarkName = bookmark  # type: ignore[attr-defined]
            flowables.append(paragraph)
        elif node.name == "p":
            flowables.append(paragraph_from_tag(node, styles["body"]))
        elif node.name in {"ul", "ol"}:
            flowables.append(list_flowable(node, styles))
        elif node.name == "pre":
            code = pdf_safe(node.get_text().rstrip())
            flowables.append(XPreformatted(html.escape(code), styles["code"]))
        elif node.name == "table":
            flowables.extend(
                [Spacer(1, 3), table_flowable(node, styles, width), Spacer(1, 7)]
            )
        elif node.name == "blockquote":
            paragraphs = [
                paragraph_from_tag(part, styles["body"])
                for part in node.find_all("p", recursive=False)
            ]
            callout = Table([[paragraphs]], colWidths=[width - 8])
            callout.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4FA")),
                        ("BOX", (0, 0), (0, -1), 1.5, colors.HexColor("#6952D5")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            flowables.extend([callout, Spacer(1, 7)])
        elif node.name == "hr":
            flowables.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#D7DDEA"),
                    spaceBefore=6,
                    spaceAfter=6,
                )
            )
    return flowables


def cover_flowables(styles: dict[str, ParagraphStyle], chapter_count: int) -> list[Any]:
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["chapter"],
        fontName="Helvetica-Bold",
        fontSize=32,
        leading=36,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#F7F8FC"),
        spaceAfter=16,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["body"],
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#CDD3E4"),
        spaceAfter=10,
    )
    label_style = ParagraphStyle(
        "CoverLabel",
        parent=styles["caption"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#7DE0DE"),
        uppercase=True,
        spaceAfter=16,
    )
    panel = Table(
        [
            [Paragraph("ZERO TO MASTERY / LOCAL LLM LAB", label_style)],
            [Paragraph("LoRA Fine-tune Studio:<br/>Zero to Mastery", title_style)],
            [Paragraph(html.escape(SUBTITLE), subtitle_style)],
            [
                Paragraph(
                    f"{chapter_count} chapters &nbsp;·&nbsp; 2 guided labs &nbsp;·&nbsp; 1 capstone",
                    label_style,
                )
            ],
        ],
        colWidths=[160 * mm],
        rowHeights=[None, None, None, None],
    )
    panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#202434")),
                ("LEFTPADDING", (0, 0), (-1, -1), 16 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16 * mm),
                ("TOPPADDING", (0, 0), (-1, 0), 14 * mm),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 14 * mm),
                ("LINEBELOW", (0, 0), (-1, 0), 1.5, colors.HexColor("#6952D5")),
            ]
        )
    )
    pipeline = Paragraph(
        "TEXT &nbsp;-&gt;&nbsp; TOKENS &nbsp;-&gt;&nbsp; VECTORS &nbsp;-&gt;&nbsp; ATTENTION &nbsp;-&gt;&nbsp; LOSS &nbsp;-&gt;&nbsp; ADAPTER",
        ParagraphStyle(
            "CoverPipeline",
            parent=styles["caption"],
            fontName="Courier-Bold",
            fontSize=8.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#503AA8"),
        ),
    )
    return [Spacer(1, 24 * mm), panel, Spacer(1, 18 * mm), pipeline]


def build_pdf(chapters: list[Chapter], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = pdf_styles()
    doc = HandbookDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=21 * mm,
        bottomMargin=17 * mm,
        title=TITLE,
        author="LoRA Fine-tune Studio",
        subject=SUBTITLE,
    )
    story: list[Any] = cover_flowables(styles, len(chapters))
    story.extend([NextPageTemplate("body"), PageBreak()])
    story.append(
        Paragraph(
            '<a name="contents"/>Contents',
            ParagraphStyle(
                "ContentsTitle",
                parent=styles["chapter"],
                fontSize=24,
                leading=28,
            ),
        )
    )
    toc = TableOfContents()
    toc.levelStyles = [
        styles["toc"],
        ParagraphStyle(
            "PDFTOC2",
            parent=styles["toc"],
            fontSize=7.7,
            leading=10.5,
            leftIndent=28,
            firstLineIndent=-10,
            textColor=colors.HexColor("#5D6475"),
        ),
    ]
    story.extend([toc, PageBreak()])
    for index, chapter in enumerate(chapters):
        if index:
            story.append(PageBreak())
        flowables = html_to_flowables(
            chapter.html, styles, doc.width, f"chapter-{chapter.number}"
        )
        if flowables:
            story.extend(flowables)
    doc.multiBuild(story)


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_bytes(files: dict[str, bytes]) -> bytes:
    manifest = {
        "source": "TUTORIAL.md",
        "files": {
            path: file_digest(data)
            for path, data in sorted(files.items())
            if path != MANIFEST_NAME
        },
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_manifest(root: Path) -> set[str]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("files", {})) | {MANIFEST_NAME}
    except OSError, ValueError, TypeError:
        return set()


def write_site(files: dict[str, bytes], root: Path) -> None:
    old_files = read_manifest(root)
    complete = dict(files)
    complete[MANIFEST_NAME] = manifest_bytes(complete)
    for relative, data in complete.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    for relative in old_files.difference(complete):
        stale = root / relative
        if stale.is_file():
            stale.unlink()


def compare_files(expected: dict[str, bytes], root: Path) -> list[str]:
    complete = dict(expected)
    complete[MANIFEST_NAME] = manifest_bytes(complete)
    differences: list[str] = []
    for relative, data in sorted(complete.items()):
        current = root / relative
        if not current.is_file():
            differences.append(f"missing: {current.relative_to(ROOT)}")
        elif current.read_bytes() != data:
            differences.append(f"stale: {current.relative_to(ROOT)}")
    for relative in sorted(read_manifest(root).difference(complete)):
        differences.append(
            f"unexpected generated file: {(root / relative).relative_to(ROOT)}"
        )
    return differences


def run(check: bool) -> int:
    _, preamble_html, chapters = load_course()
    if check:
        with tempfile.TemporaryDirectory(prefix="tutorial-build-") as temporary:
            temporary_root = Path(temporary)
            pdf_path = temporary_root / PDF_NAME
            build_pdf(chapters, pdf_path)
            pdf_bytes = pdf_path.read_bytes()
            site_files = build_site_files(preamble_html, chapters, pdf_bytes)
            differences = compare_files(site_files, DOCS_ROOT)
            canonical_pdf = PDF_ROOT / PDF_NAME
            if not canonical_pdf.is_file():
                differences.append(f"missing: {canonical_pdf.relative_to(ROOT)}")
            elif canonical_pdf.read_bytes() != pdf_bytes:
                differences.append(f"stale: {canonical_pdf.relative_to(ROOT)}")
            if differences:
                print("Tutorial outputs are not synchronized:")
                for difference in differences:
                    print(f"- {difference}")
                return 1
            print(f"Tutorial outputs are synchronized ({len(chapters)} chapters).")
            return 0

    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    canonical_pdf = PDF_ROOT / PDF_NAME
    build_pdf(chapters, canonical_pdf)
    site_files = build_site_files(preamble_html, chapters, canonical_pdf.read_bytes())
    write_site(site_files, DOCS_ROOT)
    print(f"Built {len(chapters)} HTML chapters and {canonical_pdf.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed website or PDF outputs differ from TUTORIAL.md.",
    )
    arguments = parser.parse_args()
    return run(arguments.check)


if __name__ == "__main__":
    raise SystemExit(main())
