"""Steps 2-3 - multimodal parsing -> unified JSON.   Paper Sec. II-B, Eq. (1)-(3).

PDF  -> MinerU text + PyMuPDF figures/tables (text fragments S_D, images I_D, captions C_I) Eq. (3)
CSV  -> structured records R_si = (t_i, v_i1, ... v_im)                Eq. (1) -> verbalised sentences
Both are merged into data/parsed/documents.json :  D_JSON = {(S_D, I_D, C_I)}

MinerU is used when its CLI (`mineru`, or legacy `magic-pdf`) is on PATH. MinerU v4 is
invoked through its JSON CLI and local flash tier; PyMuPDF supplies figures/tables not
included in that text-only tier. Otherwise a PyMuPDF fallback with the same output schema
is used (text blocks, image blocks, "Fig./Figure" caption detection by geometry).
"""
import csv
import html
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PARSED_IMG_DIR = config.PARSED_JSON.parent / "images"
CAP_RE = re.compile(r"^\s*(fig\.?|figure|table|图|表)\s*[\dIVX]+", re.I)
_MIN_PROSE_CHARS = 30
_MIN_ALPHA_RATIO = 0.55
_BAD_TEXT_CHARS = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]")
_PDF_LIGATURE_MAP = {
    "\ue00e": "ti", "\ue013": "ti", "\ue015": "ti", "\ue019": "ffi", "\ue01a": "fl",
    "\ue01b": "ff", "\ue01c": "ft", "\ue01d": "th", "\ue01e": "fi", "\ue01f": "Th",
    "\ue029": "ff", "\ue02b": "fl", "\ue042": "fi", "\ue043": "fl", "\ue04c": "fi",
    "\ue04d": "fl",
}
_PDF_SPECIAL_GLYPH_MAP = {"\ue001": "°", "\ue002": "•"}
_PDF_LIGATURE_RE = re.compile(f"([{re.escape(''.join(_PDF_LIGATURE_MAP))}])\\s*")


def image_url(filename: str) -> str:
    return f"{config.IMAGE_BASE_URL.rstrip('/')}/{filename}"


def _clean(t: str) -> str:
    if t is None:
        return ""
    text = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", " ", str(t)))
    text = text.replace("\ufeff", "").replace("\ufffd", " ")
    text = _PDF_LIGATURE_RE.sub(lambda match: _PDF_LIGATURE_MAP[match.group(1)], text)
    text = "".join(
        _PDF_SPECIAL_GLYPH_MAP.get(char, " " if unicodedata.category(char) == "Co" else char)
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def _normalize_document(doc: dict, source: Path, doc_id: str, modality: str) -> dict:
    """Normalize parser output and ensure every extraction unit has traceable source evidence."""
    try:
        source_path = source.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        source_path = source.name
    normalized = []
    seen = {}
    image_index = 0
    image_groups = {}
    caption_figure_ids = {}
    figure_id_aliases = {}
    for original in doc.get("units", []):
        content = _clean(original.get("content", ""))
        if not content and not original.get("url"):
            continue
        page_index = original.get("position") if modality == "pdf" else None
        page_number = page_index + 1 if isinstance(page_index, int) else None
        row_number = original.get("row_number")
        key = (original.get("modality", "Text"), content.casefold(), original.get("url"))
        if key in seen:
            previous = normalized[seen[key]]
            if page_number is not None:
                pages = previous.setdefault("duplicate_page_numbers", [previous.get("page_number")])
                if page_number not in pages:
                    pages.append(page_number)
            if row_number is not None:
                rows = previous.setdefault("duplicate_row_numbers", [previous.get("row_number")])
                if row_number not in rows:
                    rows.append(row_number)
            continue
        seen[key] = len(normalized)
        unit = dict(original)
        unit.setdefault("unit_id", f"{doc_id}:unit-{len(normalized) + 1:05d}")
        evidence = unit.get("evidence", content)
        if isinstance(evidence, str):
            evidence = _clean(evidence)
        elif isinstance(evidence, dict):
            evidence = {_clean(key): _clean(value) for key, value in evidence.items()}
        unit.update(content=content, source=source.name, source_path=source_path, doc_id=doc_id,
                    source_modality=modality, page_number=page_number, evidence=evidence)
        if unit.get("url"):
            image_index += 1
            original_figure_id = unit.setdefault(
                "figure_id",
                f"{doc_id}:p{page_number or image_index}:figure-{image_index:03d}",
            )
            caption_key = (
                page_number,
                str(unit.get("modality", "Image")).casefold(),
                content.casefold(),
            )
            figure_id = (
                caption_figure_ids.setdefault(caption_key, original_figure_id)
                if content else original_figure_id
            )
            figure_id_aliases[original_figure_id] = figure_id
            panel_index = len(image_groups.get(figure_id, {}).get("images", [])) + 1
            unit["figure_id"] = figure_id
            unit["image_id"] = f"{figure_id}:panel-{panel_index:02d}"
            unit["caption_id"] = f"{figure_id}:caption"
            unit["panel_index"] = panel_index
            unit["panel_count"] = 1
            unit.setdefault("related_text_ids", [])
            group = image_groups.setdefault(
                figure_id,
                {
                    "caption_id": unit["caption_id"],
                    "caption": content,
                    "position": original.get("position"),
                    "page_number": page_number,
                    "section": unit.get("section", ""),
                    "related_text_ids": [],
                    "images": [],
                },
            )
            if content and not group["caption"]:
                group["caption"] = content
            group["related_text_ids"].extend(unit["related_text_ids"])
            group["images"].append(unit)
            for panel_unit in group["images"]:
                panel_unit["panel_count"] = len(group["images"])
            unit["content"] = ""
        if "section" in unit:
            unit["section"] = _clean(unit.get("section"))
        normalized.append(unit)

    for figure_id, group in image_groups.items():
        image_ids = [unit["image_id"] for unit in group["images"]]
        image_urls = [unit["url"] for unit in group["images"]]
        normalized.append({
            "unit_id": group["caption_id"],
            "content": group["caption"],
            "modality": "Caption",
            "position": group["position"],
            "page_number": group["page_number"],
            "section": group["section"],
            "source": source.name,
            "source_path": source_path,
            "doc_id": doc_id,
            "source_modality": modality,
            "evidence": group["caption"],
            "figure_id": figure_id,
            "caption_id": group["caption_id"],
            "image_ids": image_ids,
            "image_urls": image_urls,
            "image_url": image_urls[0] if image_urls else None,
            "related_text_ids": sorted(set(group["related_text_ids"])),
            "related_figure_ids": [figure_id],
        })

    doc["units"] = normalized

    valid_text_ids = {
        unit["unit_id"]
        for unit in normalized
        if unit.get("modality", "").casefold() == "text"
    }
    for image_index, unit in enumerate(normalized):
        if not unit.get("url"):
            continue
        group = image_groups[unit["figure_id"]]
        group["related_text_ids"] = [
            text_id for text_id in group["related_text_ids"] if text_id in valid_text_ids
        ]
        if not group["related_text_ids"]:
            nearby_text = sorted(
                (
                    (abs(text_index - image_index), text_unit["unit_id"])
                    for text_index, text_unit in enumerate(normalized)
                    if text_unit["unit_id"] in valid_text_ids
                    and text_unit.get("page_number") == unit.get("page_number")
                ),
                key=lambda candidate: candidate[0],
            )
            group["related_text_ids"].extend(text_id for _, text_id in nearby_text[:2])
        unit["related_text_ids"] = sorted(set(group["related_text_ids"]))
        for related_id in unit["related_text_ids"]:
            related = next((entry for entry in normalized if entry["unit_id"] == related_id), None)
            if related is not None:
                figures = {
                    figure_id_aliases.get(related_figure_id, related_figure_id)
                    for related_figure_id in related.get("related_figure_ids", [])
                }
                figures.add(unit["figure_id"])
                related["related_figure_ids"] = sorted(figures)
    for figure_id, group in image_groups.items():
        caption_unit = next(unit for unit in normalized if unit["unit_id"] == group["caption_id"])
        caption_unit["related_text_ids"] = sorted(set(group["related_text_ids"]))

    units_by_id = {unit["unit_id"]: unit for unit in normalized}
    captions_by_id = {
        unit["unit_id"]: unit["content"]
        for unit in normalized
        if unit.get("modality", "").casefold() == "caption"
    }
    doc["image_records"] = [
        {
            "image_id": unit["image_id"],
            "figure_id": unit["figure_id"],
            "caption_id": unit["caption_id"],
            "panel_index": unit["panel_index"],
            "panel_count": unit["panel_count"],
            "url": unit["url"],
            "caption": captions_by_id.get(unit["caption_id"], ""),
            "source": unit["source"],
            "source_path": unit["source_path"],
            "page_number": unit["page_number"],
            "related_text_ids": unit["related_text_ids"],
            "related_text": [
                {
                    "unit_id": related_id,
                    "text": units_by_id[related_id]["content"],
                }
                for related_id in unit["related_text_ids"]
                if related_id in units_by_id
            ],
        }
        for unit in normalized
        if unit.get("url")
    ]
    doc["metadata"] = {
        "doc_id": doc_id,
        "source": source.name,
        "source_path": source_path,
        "modality": modality,
        "parser": doc.get("parser", "unknown"),
        "unit_count": len(normalized),
        "encoding": "utf-8-normalized",
    }
    return doc


def _pdf_text_rejection(text: str) -> str:
    """Return a reason for discarding an obviously incomplete or corrupted prose fragment."""
    if _BAD_TEXT_CHARS.search(str(text or "")):
        return "replacement/control characters"
    text = _clean(text)
    if len(text) < _MIN_PROSE_CHARS:
        return "short fragment"
    if re.fullmatch(r"(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/[\w./-]*)?", text, re.I):
        return "page URL"
    words = re.findall(r"[A-Za-z]+", text)
    if (
        len(words) >= 6
        and sum(word[0].isupper() for word in words) / len(words) >= 0.8
        and re.search(r"(?:,|&)\s*[A-Z]", text)
        and not re.search(r"[.!?;:]", text)
    ):
        return "author byline"
    if _BAD_TEXT_CHARS.search(text):
        return "replacement/control characters"
    letters = sum(char.isalpha() for char in text)
    if letters / len(text) < _MIN_ALPHA_RATIO:
        return "low alphabetic content"
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) >= 5 and sum(len(word) == 1 for word in words) / len(words) >= 0.4:
        return "fragmented words"
    return ""


def _furniture_key(text: str):
    text = re.sub(r"\d+", "#", _clean(text).casefold())
    return text if len(text) <= 160 else ""


def _repeated_furniture(entries, page_count: int):
    """Find short text repeated on multiple pages, optionally restricted to a page region."""
    if page_count < 3:
        return set()
    pages_by_key = {}
    for page, region, text in entries:
        key = (region, _furniture_key(text))
        if key[1]:
            pages_by_key.setdefault(key, set()).add(page)
    threshold = max(3, math.ceil(page_count * 0.25))
    return {key for key, pages in pages_by_key.items() if len(pages) >= threshold}


# ------------------------------------------------------------------ MinerU
def _html_table_rows(markup: str):
    """Extract header/data rows from MinerU's HTML table representation."""
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.row = None
            self.cell = None

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.cell = []

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.cell is not None:
                self.row.append(_clean(" ".join(self.cell)))
                self.cell = None
            elif tag == "tr" and self.row is not None:
                if any(self.row):
                    self.rows.append(self.row)
                self.row = None

    parser = TableParser()
    parser.feed(markup or "")
    return parser.rows


def mineru_available() -> str:
    return shutil.which("mineru") or shutil.which("magic-pdf") or ""


def _run_mineru_command(cmd, timeout):
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MinerU command timed out after {timeout} seconds.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run MinerU command: {exc}") from exc


def _is_mineru_v4(exe: str) -> bool:
    result = _run_mineru_command([exe, "--version"], timeout=30)
    version = result.stdout + "\n" + result.stderr
    match = re.search(r"MinerU version:\s*(\d+)", version, re.I)
    return bool(match and int(match.group(1)) >= 4)


def _mineru_v4_parse_payload(exe: str, pdf: Path):
    command = [
        exe, "parse", str(pdf), "--tier", "flash", "--pages", "all",
        "--json", "--wait", "1800",
    ]

    def parse_response():
        result = _run_mineru_command(command, timeout=1830)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"MinerU v4 returned invalid JSON: {detail[-500:] or 'no diagnostic output'}"
            ) from exc

        error = payload.get("error")
        if error:
            code = error.get("code", "unknown_error")
            message = error.get("message", "MinerU reported an unspecified error.")
            if code == "server_not_running":
                start = _run_mineru_command([exe, "server", "start"], timeout=180)
                if start.returncode != 0:
                    detail = (start.stderr or start.stdout).strip()
                    raise RuntimeError(
                        f"Could not start the local MinerU server: {detail[-500:] or start.returncode}"
                    )
                result = _run_mineru_command(command, timeout=1830)
                try:
                    payload = json.loads(result.stdout)
                except json.JSONDecodeError as exc:
                    detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(
                        f"MinerU v4 returned invalid JSON after server startup: "
                        f"{detail[-500:] or 'no diagnostic output'}"
                    ) from exc
                error = payload.get("error")
                if not error:
                    return payload
                code = error.get("code", "unknown_error")
                message = error.get("message", "MinerU reported an unspecified error.")
            raise RuntimeError(f"MinerU v4 parse failed ({code}): {message}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"MinerU v4 exited with code {result.returncode}: {detail[-500:]}")
        return payload

    return parse_response()


def _markdown_table_rows(lines):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _parse_mineru_v4_markdown(markdown: str, doc_id: str):
    """Adapt MinerU v4's flash-tier Markdown response to the project's parser schema."""
    S, units, records, formulas, sections = [], [], [], [], []
    page_index = 0
    filtered_text = 0
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    i = 0

    def add_text(text, modality="Text"):
        nonlocal filtered_text
        text = _clean(text)
        if not text:
            return
        reason = _pdf_text_rejection(text) if modality == "Text" else ""
        if reason:
            filtered_text += 1
            return
        if modality == "Text":
            S.append(text)
        elif modality == "Formula":
            formulas.append(text)
        units.append(dict(content=text, modality=modality, position=page_index))

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        page_marker = re.match(r"<!--\s*page\s+(\d+)\s+of\s+\d+\s*-->", line, re.I)
        if page_marker:
            page_index = max(0, int(page_marker.group(1)) - 1)
            i += 1
            continue

        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            title = _clean(heading.group(1))
            sections.append(dict(title=title, page=page_index))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = _markdown_table_rows(table_lines)
            table_records = _row_sentences("Table", rows)
            for record in table_records:
                records.append(record["content"])
                units.append(dict(record, modality="TableRow", position=page_index))
            continue

        if line.startswith("$$") or line.startswith(r"\["):
            formula_lines = [line]
            closing = "$$" if line.startswith("$$") else r"\]"
            if not line.endswith(closing) or len(line) == len(closing):
                i += 1
                while i < len(lines):
                    formula_lines.append(lines[i].strip())
                    if closing in lines[i]:
                        break
                    i += 1
            add_text(" ".join(formula_lines), "Formula")
            i += 1
            continue

        paragraph = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if (not next_line or next_line.startswith("|")
                    or re.match(r"<!--\s*page\s+\d+\s+of\s+\d+\s*-->", next_line, re.I)
                    or re.match(r"^#{1,6}\s+", next_line)):
                break
            paragraph.append(next_line)
            i += 1
        text = " ".join(paragraph)
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*`~]", "", text)
        text = _clean(text)
        byline_break = re.match(r"^(.{30,}?)\s+\\\s+(.+)$", text)
        if byline_break and _pdf_text_rejection(byline_break.group(1)) == "author byline":
            text = byline_break.group(2)
        add_text(text)

    return dict(S_D=S, I_D=[], C_I=[], units=units, records=records, formulas=formulas, sections=sections,
                parser="mineru-v4-flash", mineru_tier="flash", filtered_text_fragments=filtered_text)


def parse_pdf_mineru(pdf: Path, doc_id: str):
    exe = mineru_available()
    out = config.MINERU_OUT_DIR / doc_id
    out.mkdir(parents=True, exist_ok=True)
    if _is_mineru_v4(exe):
        payload = _mineru_v4_parse_payload(exe, pdf)
        content = payload.get("content")
        markdown = content.get("content") if isinstance(content, dict) else None
        if not isinstance(markdown, str):
            raise RuntimeError("MinerU v4 completed without returning Markdown content.")
        (out / "mineru_v4_response.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        parsed = _parse_mineru_v4_markdown(markdown, doc_id)
        visual_data = parse_pdf_pymupdf(pdf, doc_id)
        parsed["I_D"] = visual_data["I_D"]
        parsed["C_I"] = visual_data["C_I"]
        parsed["records"] = visual_data.get("records", [])
        parsed["formulas"] = list(dict.fromkeys(
            parsed["formulas"] + visual_data.get("formulas", [])
        ))
        if not parsed["sections"]:
            parsed["sections"] = visual_data.get("sections", [])
        parsed["units"].extend(
            unit for unit in visual_data.get("units", []) if unit.get("modality") != "Text"
        )
        parsed["filtered_text_fragments"] += visual_data.get("filtered_text_fragments", 0)
        parsed["parser"] = "mineru-v4-flash+pymupdf-assets"
        return parsed

    cmd = [exe, "-p", str(pdf), "-o", str(out)]
    result = subprocess.run(cmd, check=False, capture_output=True, timeout=1800)
    if result.returncode:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Legacy MinerU CLI exited with code {result.returncode}: {detail[-500:]}")
    cl = next(iter(out.rglob("*_content_list.json")), None)
    if cl is None:
        raise RuntimeError("MinerU produced no content_list.json")
    items = json.loads(cl.read_text(encoding="utf-8"))
    S, I, C, units, records = [], [], [], [], []
    page_count = max((it.get("page_idx", 0) for it in items), default=0) + 1
    furniture = _repeated_furniture(
        ((it.get("page_idx", 0), "any", it.get("text", ""))
         for it in items if it.get("type") == "text"),
        page_count,
    )
    filtered_text = 0
    PARSED_IMG_DIR.mkdir(parents=True, exist_ok=True)
    figure_counts = {}

    def add_mineru_figure(src, caption, page_index, modality, split_panels):
        figure_counts[page_index] = figure_counts.get(page_index, 0) + 1
        figure_id = f"{doc_id}:p{page_index + 1}:figure-{figure_counts[page_index]:03d}"
        data = src.read_bytes()
        panels = _image_panels(data) if split_panels else [(data, False)]
        for panel_index, (panel_data, was_split) in enumerate(panels, 1):
            suffix = "png" if was_split else src.suffix.lstrip(".")
            fname = f"{doc_id}_{len(I):02d}.{suffix}"
            (PARSED_IMG_DIR / fname).write_bytes(panel_data)
            url = image_url(fname)
            image_id = f"{figure_id}:panel-{panel_index:02d}"
            I.append(url)
            C.append(caption)
            units.append(dict(
                content=caption, modality=modality, position=page_index, url=url,
                image_id=image_id, figure_id=figure_id, caption_id=f"{figure_id}:caption",
                panel_index=panel_index, panel_count=len(panels), related_text_ids=[],
            ))

    for it in items:
        typ = it.get("type")
        if typ == "text" and it.get("text", "").strip():
            text = _clean(it["text"])
            reason = _pdf_text_rejection(text)
            if reason or ("any", _furniture_key(text)) in furniture:
                filtered_text += 1
                continue
            S.append(text)
            units.append(dict(content=S[-1], modality="Text", position=it.get("page_idx", 0)))
        elif typ == "equation" and it.get("text", "").strip():
            units.append(dict(content=_clean(it["text"]), modality="Formula", position=it.get("page_idx", 0)))
        elif typ == "table":
            caps = it.get("table_caption") or it.get("img_caption") or []
            caps = caps if isinstance(caps, list) else [caps]
            cap = _clean(" ".join(str(item) for item in caps if item))
            row_data = _html_table_rows(it.get("table_body", ""))
            table_records = _row_sentences(cap or "Table", row_data)
            for record in table_records:
                records.append(record["content"])
                units.append(dict(record, modality="TableRow", position=it.get("page_idx", 0),
                                  table_caption=cap))
            table_image_added = False
            if it.get("img_path"):
                src = cl.parent / it["img_path"]
                if src.exists() and (cap or not config.IMAGE_REQUIRE_CAPTION):
                    add_mineru_figure(src, cap, it.get("page_idx", 0), "Table", split_panels=False)
                    table_image_added = True
            if cap and not table_image_added and not table_records:
                units.append(dict(content=cap, modality="Table", position=it.get("page_idx", 0),
                                  table_caption=cap))
        elif typ == "image" and it.get("img_path"):
            src = cl.parent / it["img_path"]
            if not src.exists():
                continue
            caps = it.get("img_caption") or it.get("image_caption") or []
            caps = caps if isinstance(caps, list) else [caps]
            cap = _clean(" ".join(str(item) for item in caps if item)) if caps else ""
            if cap or not config.IMAGE_REQUIRE_CAPTION:
                add_mineru_figure(src, cap, it.get("page_idx", 0), "Image", split_panels=True)
    return dict(S_D=S, I_D=I, C_I=C, units=units, records=records, formulas=[], parser="mineru",
                filtered_text_fragments=filtered_text)


# ---------------------------------------------------------------- fallback
TABLE_CAP_RE = re.compile(r"^\s*(table|表)\s*[\dIVX]+", re.I)
FIG_CAP_RE = re.compile(r"^\s*(fig\.?|figure|图)\s*[\dIVXA]+", re.I)
EQ_RE = re.compile(r"=.*\(\d+\)\s*$")          # numbered equation line:  "R = ... (1)"


def _row_sentences(cap: str, rows):
    """Table -> one sentence per data row ("Table 1 ... row: header = value; ...") so the numbers reach the RAG / LLM."""
    rows = [[_clean(str(c or "")) for c in r] for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return []
    head, label = rows[0], re.match(r"^\s*((?:table|表)\s*[\dIVX]+)", cap, re.I)
    tag = label.group(1) if label else "Table"
    short = re.sub(r"^\s*(?:table|表)\s*[\dIVX]+[.:]?\s*", "", cap, flags=re.I).rstrip(".")
    return [
        dict(
            content=f"{tag} ({short}), row: " + "; ".join(
                f"{h} = {v}" for h, v in zip(head, r) if v
            ) + ".",
            row_number=index,
            evidence=dict(zip(head, r)),
        )
        for index, r in enumerate(rows[1:], 1)
        if any(r)
    ]


def _ocr_page_text(page):
    """OCR a scanned page when Tesseract is installed; otherwise report it as unavailable."""
    if not shutil.which("tesseract"):
        return None
    textpage = page.get_textpage_ocr(language="eng", dpi=200, full=True)
    return _clean(page.get_text("text", textpage=textpage))


def _image_panels(image_data: bytes):
    """Split a composite figure only at clear, full-length white gutters."""
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except (OSError, ValueError) as exc:
        print(f"[parser] Could not inspect embedded image for panels ({type(exc).__name__}); keeping it whole.")
        return [(image_data, False)]

    pixels = np.asarray(image)
    white = np.all(pixels >= 245, axis=2)
    cuts = {}
    for axis, projection in (
        ("x", white.mean(axis=0)),
        ("y", white.mean(axis=1)),
    ):
        min_gap = max(3, round(len(projection) * 0.002))
        max_gap = max(min_gap, round(len(projection) * 0.05))
        min_panel = round(len(projection) * 0.15)
        gaps = []
        start = None
        for index, is_white in enumerate(projection >= 0.98):
            if is_white and start is None:
                start = index
            elif not is_white and start is not None:
                if min_gap <= index - start <= max_gap:
                    gaps.append((start, index))
                start = None
        if start is not None and min_gap <= len(projection) - start <= max_gap:
            gaps.append((start, len(projection)))

        segments = []
        cursor = 0
        for start, end in gaps:
            if start - cursor >= min_panel and len(projection) - end >= min_panel:
                segments.append((cursor, start))
                cursor = end
        segments.append((cursor, len(projection)))
        if len(segments) > 1:
            cuts[axis] = segments
        else:
            cuts[axis] = [(0, len(projection))]

    regions = [
        (left, top, right, bottom)
        for top, bottom in cuts["y"]
        for left, right in cuts["x"]
    ]
    if len(regions) <= 1:
        return [(image_data, False)]

    panels = []
    for region in regions:
        output = BytesIO()
        image.crop(region).save(output, format="PNG")
        panels.append((output.getvalue(), True))
    return panels


def parse_pdf_pymupdf(pdf: Path, doc_id: str):
    """Fallback parser.  Produces the same schema as MinerU:
      S_D text fragments, I_D image URLs + C_I captions (figures AND tables-as-images, Sec. III-C),
      records (one sentence per table row), formulas (numbered equations), sections (headings)."""
    import statistics
    import pymupdf
    PARSED_IMG_DIR.mkdir(parents=True, exist_ok=True)
    S, I, C, units, records, formulas, sections = [], [], [], [], [], [], []
    doc = pymupdf.open(str(pdf))

    sizes = []                                              # body font size = most common size, weighted by characters
    furniture_entries = []
    for pno, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    sizes += [round(sp["size"], 1)] * len(sp["text"].strip())
            text = _clean(" ".join(
                sp["text"] for line in b.get("lines", []) for sp in line["spans"]
            ))
            if text:
                y0, y1 = b["bbox"][1], b["bbox"][3]
                if y0 <= page.rect.height * 0.08:
                    furniture_entries.append((pno, "top", text))
                elif y1 >= page.rect.height * 0.92:
                    furniture_entries.append((pno, "bottom", text))
    repeated_furniture = _repeated_furniture(furniture_entries, len(doc))
    body = statistics.mode(sizes) if sizes else 10.0
    cur_section = ""
    seen_img = set()                                        # repeated logos / page headers are kept once
    filtered_text = 0
    ocr_pages = []
    ocr_unavailable_pages = []

    for pno, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        texts = []                                          # (bbox, text, size, bold)
        text_ids = []
        for b in blocks:
            if b["type"] != 0:
                continue
            spans = [sp for l in b["lines"] for sp in l["spans"] if sp["text"].strip()]
            t = _clean(" ".join(sp["text"] for sp in spans))
            if t:
                bold = all((sp["flags"] & 16) or "bold" in sp["font"].lower() for sp in spans)
                texts.append((b["bbox"], t, max(sp["size"] for sp in spans), bold))
                text_ids.append(f"{doc_id}:p{pno + 1}:text-{len(text_ids) + 1:03d}")
        if not texts:
            ocr_text = _ocr_page_text(page)
            if ocr_text is None:
                ocr_unavailable_pages.append(pno + 1)
            elif ocr_text:
                reason = _pdf_text_rejection(ocr_text)
                if reason:
                    filtered_text += 1
                else:
                    S.append(ocr_text)
                    units.append(dict(content=ocr_text, modality="Text", position=pno,
                                      section=cur_section, extraction_method="tesseract_ocr"))
                    ocr_pages.append(pno + 1)
        used = set()
        related_figures_by_text_id = {}
        page_figure_index = 0

        def add_figure(
            image_data, image_rect, caption, modality="Image", section="", image_ext="png",
            split_panels=True,
        ):
            nonlocal page_figure_index
            page_figure_index += 1
            figure_id = f"{doc_id}:p{pno + 1}:figure-{page_figure_index:03d}"
            caption_index = next(
                (
                    index for index, (text_box, text, *_rest) in enumerate(texts)
                    if text == caption and text_box[1] >= image_rect.y1 - 5
                ),
                None,
            )
            related_candidates = []
            for index, (text_box, text, *_rest) in enumerate(texts):
                if index == caption_index or len(text) < 30:
                    continue
                if CAP_RE.match(text):
                    continue
                box = pymupdf.Rect(text_box)
                rect = pymupdf.Rect(image_rect)
                overlap = max(0, min(box.x1, rect.x1) - max(box.x0, rect.x0))
                overlap_ratio = overlap / max(1, min(box.width, rect.width))
                if overlap_ratio < 0.35:
                    continue
                if box.y1 <= rect.y0:
                    gap, direction = rect.y0 - box.y1, "above"
                elif box.y0 >= rect.y1:
                    gap, direction = box.y0 - rect.y1, "below"
                else:
                    gap, direction = 0, "overlap"
                if gap <= 90:
                    related_candidates.append((gap, direction, index))
            related_text_ids = []
            for direction in ("overlap", "above", "below"):
                directional = sorted(
                    (candidate for candidate in related_candidates if candidate[1] == direction),
                    key=lambda candidate: candidate[0],
                )
                for _, _, index in directional[:1]:
                    text_id = text_ids[index]
                    related_text_ids.append(text_id)
                    related_figures_by_text_id.setdefault(text_id, set()).add(figure_id)

            panel_bytes = _image_panels(image_data) if split_panels else [(image_data, False)]
            panel_count = len(panel_bytes)
            for panel_index, (panel, was_split) in enumerate(panel_bytes, 1):
                suffix = "png" if was_split else image_ext
                fname = f"{doc_id}_{len(I):02d}.{suffix}"
                (PARSED_IMG_DIR / fname).write_bytes(panel)
                url = image_url(fname)
                image_id = f"{figure_id}:panel-{panel_index:02d}"
                I.append(url)
                C.append(caption)
                units.append(dict(
                    content=caption,
                    modality=modality,
                    position=pno,
                    url=url,
                    section=section,
                    image_id=image_id,
                    figure_id=figure_id,
                    caption_id=f"{figure_id}:caption",
                    panel_index=panel_index,
                    panel_count=panel_count,
                    related_text_ids=related_text_ids,
                ))

        # ---- tables  -> image (paper treats tables as images) + row sentences
        table_boxes = []
        try:
            found = page.find_tables().tables
        except Exception:  # noqa: BLE001
            found = []
        for tb in found:
            rows = tb.extract()
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            bb = pymupdf.Rect(tb.bbox)
            cap, ci = "", None
            above = [(i, x) for i, x in enumerate(texts) if x[0][3] <= bb.y0 + 2 and x[0][3] >= bb.y0 - 60]
            if above:
                i, x = max(above, key=lambda t: t[1][0][3])
                if TABLE_CAP_RE.match(x[1]):
                    cap, ci = x[1], i
            if ci is None:
                below = [(i, x) for i, x in enumerate(texts) if x[0][1] >= bb.y1 - 2 and x[0][1] <= bb.y1 + 40 and TABLE_CAP_RE.match(x[1])]
                if below:
                    ci, cap = below[0][0], below[0][1][1]
            if ci is not None:
                used.add(ci)
            if cap or not config.IMAGE_REQUIRE_CAPTION:      # uncaptioned table images cannot join the graph
                pixmap = page.get_pixmap(clip=(bb + (-4, -4, 4, 4)) & page.rect, dpi=150)
                add_figure(pixmap.tobytes("png"), bb, cap, modality="Table", section=cur_section,
                           split_panels=False)
            table_records = _row_sentences(cap or "Table", rows)
            records += [record["content"] for record in table_records]
            units.extend(dict(record, modality="TableRow", position=pno, section=cur_section,
                              table_caption=cap) for record in table_records)
            table_boxes.append(bb)
            for j, x in enumerate(texts):                    # cell text is represented by the rows, not as prose
                if pymupdf.Rect(x[0]).intersects(bb) and j != ci:
                    used.add(j)

        # ---- embedded raster figures with the caption below
        for b in sorted([b for b in blocks if b["type"] == 1], key=lambda b: b["bbox"][1]):
            if min(b.get("width", 0), b.get("height", 0)) < config.MIN_IMAGE_PX:
                continue                                                      # icon / logo / decoration
            key = (pno, hash(b["image"]), tuple(round(value, 1) for value in b["bbox"]))
            if key in seen_img:
                continue
            seen_img.add(key)
            below = [
                (i, x) for i, x in enumerate(texts)
                if x[0][1] >= b["bbox"][3] - 5
                and x[0][1] - b["bbox"][3] <= 90
            ]
            below.sort(key=lambda x: x[1][0][1])
            cap = ""
            pick = next((x for x in below[:2] if FIG_CAP_RE.match(x[1][1])), None)
            if pick:
                cap = pick[1][1]
                used.add(pick[0])
            if not cap and config.IMAGE_REQUIRE_CAPTION:
                continue

            add_figure(b["image"], pymupdf.Rect(b["bbox"]), cap, section=cur_section,
                       image_ext=b.get("ext", "png"))

        # ---- vector figures (charts drawn with paths); table grid lines are excluded
        drawings = [d["rect"] for d in page.get_drawings() if (d["rect"].width > 2 or d["rect"].height > 2)
                    and not any(tb.contains(d["rect"]) or tb.intersects(d["rect"]) for tb in table_boxes)]
        for i, (bb, t, _, _) in enumerate(texts):
            if i in used or not FIG_CAP_RE.match(t):
                continue
            above = sorted([r for r in drawings if r.y1 <= bb[3] and r.y1 >= bb[1] - 450], key=lambda r: -r.y1)
            if not above:
                continue
            x0, y0, x1, y1 = above[0].x0, above[0].y0, above[0].x1, above[0].y1
            for r in above[1:]:                              # grow upward while gaps stay small (manual union: PyMuPDF drops zero-width lines)
                if r.y1 >= y0 - 40:
                    x0, y0, x1, y1 = min(x0, r.x0), min(y0, r.y0), max(x1, r.x1), max(y1, r.y1)
            if (y1 - y0) < 60 or (x1 - x0) < 100:
                continue
            figure_rect = pymupdf.Rect(x0, y0, x1, y1)
            nearby_text = pymupdf.Rect(
                figure_rect.x0 - 20, figure_rect.y0 - 20,
                figure_rect.x1 + 20, figure_rect.y1 + 20,
            )
            for text_index, (text_box, text, *_rest) in enumerate(texts):
                if text_index == i or len(text) > 80 or FIG_CAP_RE.match(text):
                    continue
                label_rect = pymupdf.Rect(text_box)
                if nearby_text.intersects(label_rect):
                    figure_rect |= label_rect
            cl = pymupdf.Rect(
                figure_rect.x0 - 4, figure_rect.y0 - 4,
                figure_rect.x1 + 4, min(bb[1] - 2, figure_rect.y1 + 4),
            ) & page.rect
            pixmap = page.get_pixmap(clip=cl, dpi=150)
            add_figure(pixmap.tobytes("png"), cl, t, section=cur_section, split_panels=False)
            used.add(i)
            for j, (b2, t2, *_r) in enumerate(texts):         # axis labels / ticks inside the figure are not prose
                if j not in used and len(t2) < 60 and cl.contains(pymupdf.Rect(b2)):
                    used.add(j)

        # ---- headings (sections), formulas, running text
        for i, (bb, t, size, bold) in enumerate(texts):
            if i in used:
                continue
            y0, y1 = bb[1], bb[3]
            region = "top" if y0 <= page.rect.height * 0.08 else (
                "bottom" if y1 >= page.rect.height * 0.92 else ""
            )
            if region and (region, _furniture_key(t)) in repeated_furniture:
                filtered_text += 1
                continue
            is_head = (size >= body * 1.12 or (bold and size >= body * 0.98)) and len(t) < 100 and not t.endswith(".")
            if is_head:
                if sections and sections[-1]["page"] == pno and abs(sections[-1]["size"] - size) < 0.1 and 0 <= bb[1] - sections[-1]["y1"] < 10:
                    sections[-1]["title"] += " " + t                # wrapped title: join the lines
                    sections[-1]["y1"] = bb[3]
                    cur_section = sections[-1]["title"]
                else:
                    cur_section = t
                    sections.append(dict(title=t, page=pno, size=size, y1=bb[3]))
                continue
            if EQ_RE.search(t) and len(t) < 160:
                formulas.append(t)
                units.append(dict(content=t, modality="Formula", position=pno, section=cur_section,
                                  unit_id=text_ids[i],
                                  related_figure_ids=sorted(related_figures_by_text_id.get(text_ids[i], set()))))
            else:
                reason = _pdf_text_rejection(t)
                if reason:
                    filtered_text += 1
                    continue
                units.append(dict(content=t, modality="Text", position=pno, section=cur_section,
                                  unit_id=text_ids[i],
                                  related_figure_ids=sorted(related_figures_by_text_id.get(text_ids[i], set()))))
            S.append(t)
    for sec in sections:
        sec.pop("size", None), sec.pop("y1", None)
    result = dict(S_D=S, I_D=I, C_I=C, units=units, records=records, formulas=formulas, sections=sections,
                  parser="pymupdf-fallback", filtered_text_fragments=filtered_text,
                  ocr_pages=ocr_pages, ocr_unavailable_pages=ocr_unavailable_pages)
    doc.close()
    return result


# -------------------------------------------------------------------- HTML
def parse_html(path: Path, doc_id: str):
    """Semi-structured source: headings -> sections, <p> -> text, <table> -> row sentences + caption,
    <figure>/<figcaption> -> image + caption, class="formula" -> formula."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.buf = [], ""
            self.S, self.I, self.C, self.units, self.records, self.formulas, self.sections = [], [], [], [], [], [], []
            self.sec, self.rows, self.row, self.tcap, self.imgs, self.fig_src = "", None, None, "", [], None
            self.filtered_text = 0

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            self.stack.append((tag, a.get("class", "")))
            self.buf = "" if tag in ("p", "h1", "h2", "h3", "figcaption", "caption", "th", "td", "div") else self.buf
            if tag == "table":
                self.rows, self.tcap = [], ""
            elif tag == "tr" and self.rows is not None:
                self.row = []
            elif tag == "img" and a.get("src"):
                self.fig_src = a["src"]

        def handle_data(self, d):
            self.buf += d

        def handle_endtag(self, tag):
            t = _clean(self.buf)
            cls = self.stack[-1][1] if self.stack else ""
            if self.stack:
                self.stack.pop()
            if tag in ("h1", "h2", "h3") and t:
                self.sec = t
                self.sections.append(dict(title=t, page=0))
            elif tag == "p" and t:
                if _pdf_text_rejection(t):
                    self.filtered_text += 1
                else:
                    self.S.append(t)
                    self.units.append(dict(content=t, modality="Text", position=0, section=self.sec))
            elif tag == "div" and "formula" in cls and t:
                self.formulas.append(t)
                self.S.append(t)
                self.units.append(dict(content=t, modality="Formula", position=0, section=self.sec))
            elif tag == "caption":
                self.tcap = t
            elif tag in ("th", "td") and self.row is not None:
                self.row.append(t)
            elif tag == "tr" and self.rows is not None and self.row:
                self.rows.append(self.row)
                self.row = None
            elif tag == "table" and self.rows:
                table_records = _row_sentences(self.tcap or "Table", self.rows)
                self.records += [record["content"] for record in table_records]
                self.units.extend(dict(record, modality="TableRow", position=0, section=self.sec,
                                       table_caption=self.tcap) for record in table_records)
                if self.tcap:
                    self.units.append(dict(content=self.tcap, modality="Table", position=0, section=self.sec))
                self.rows = None
            elif tag == "figcaption" and t:
                src = (path.parent / self.fig_src).resolve() if self.fig_src else None
                if src and src.exists():
                    PARSED_IMG_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, PARSED_IMG_DIR / src.name)
                    self.I.append(image_url(src.name))
                    self.C.append(t)
                    self.units.append(dict(content=t, modality="Image", position=0, url=self.I[-1], section=self.sec))
                self.fig_src = None
            self.buf = ""

    p = P()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))
    return dict(S_D=p.S, I_D=p.I, C_I=p.C, units=p.units, records=p.records, formulas=p.formulas, sections=p.sections,
                parser="html", filtered_text_fragments=p.filtered_text)


def parse_pdf(pdf: Path, doc_id: str):
    if mineru_available():
        try:
            return parse_pdf_mineru(pdf, doc_id)
        except Exception as e:  # noqa: BLE001
            detail = " ".join(str(e).split())
            print(f"[parser] MinerU failed on {pdf.name} ({type(e).__name__}: {detail[:400]}); "
                  "using PyMuPDF fallback")
    return parse_pdf_pymupdf(pdf, doc_id)


# --------------------------------------------------------------------- CSV
def record_to_sentence(rec: dict, cols):
    t = rec.get(cols[0], "")
    vals = {c: rec.get(c, "").strip() for c in cols[1:]}
    proc, eq = vals.get("Process", ""), vals.get("Equipment", "")
    status, prob = vals.get("Operating status", ""), vals.get("Problem description", "")
    s = f"At {t}, the {proc} process equipment {eq} reported: {status}."
    if prob and prob.lower() != "no abnormalities":
        s += f" Problem: {prob}."
    else:
        s += " No abnormalities were found."
    return s


def generic_record_to_sentence(rec: dict, cols):
    """Any other tabular plant log: first column = timestamp, rest = attribute values (Eq. 1)."""
    t = rec.get(cols[0], "")
    kv = ", ".join(f"{c.strip()} is {str(rec.get(c, '')).strip()}" for c in cols[1:] if str(rec.get(c, "")).strip())
    return f"At {t}, the flotation plant recorded: {kv}."


def parse_csv(path: Path, doc_id: str):
    """D_stru = {R_si | R_si = (t_i, v_i1 ... v_im)}  -> records + verbalised sentences.
    Delimiter is sniffed (',' or ';'); very long logs are sub-sampled evenly to config.CSV_MAX_ROWS rows."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    raw = raw.replace("\ufffd", " ")
    lines = raw.splitlines()
    header = lines[0] if lines else ""
    delim = max(",;\t", key=header.count)          # header has no quoted numbers, so counting is reliable
    rows = list(csv.DictReader(lines, delimiter=delim))
    rows = [{_clean(key): _clean(value) for key, value in row.items() if key is not None}
            for row in rows]
    if len(rows) > config.CSV_MAX_ROWS:
        step = len(rows) / config.CSV_MAX_ROWS
        rows = [rows[int(i * step)] for i in range(config.CSV_MAX_ROWS)]
    cols = list(rows[0].keys()) if rows else []
    inspection = {"Process", "Equipment", "Operating status"} <= set(cols)
    to_sentence = record_to_sentence if inspection else generic_record_to_sentence
    records = [to_sentence(r, cols) for r in rows]
    units = [dict(content=sentence, modality="TableRow", position=i, row_number=i + 1,
                  evidence=rows[i], table_caption=path.stem)
             for i, sentence in enumerate(records) if sentence.strip()]
    return dict(S_D=[], I_D=[], C_I=[], units=units,
                records=records, structured=rows, parser="csv")


# -------------------------------------------------------------------- main
def run():
    docs = []
    for pdf in sorted(config.RAW_PDF_DIR.glob("*.pdf")):
        doc_id = re.sub(r"\W+", "_", pdf.stem)
        d = _normalize_document(parse_pdf(pdf, doc_id), pdf, doc_id, "pdf")
        docs.append(dict(doc_id=doc_id, source=pdf.name, modality="pdf", **d))
    for htmlf in sorted(list(config.RAW_HTML_DIR.glob("*.html")) + list(config.RAW_HTML_DIR.glob("*.htm"))):
        doc_id = re.sub(r"\W+", "_", htmlf.stem)
        d = _normalize_document(parse_html(htmlf, doc_id), htmlf, doc_id, "html")
        docs.append(dict(doc_id=doc_id, source=htmlf.name, modality="html", **d))
    for csvf in sorted(config.RAW_CSV_DIR.glob("*.csv")):
        doc_id = re.sub(r"\W+", "_", csvf.stem)
        d = _normalize_document(parse_csv(csvf, doc_id), csvf, doc_id, "csv")
        docs.append(dict(doc_id=doc_id, source=csvf.name, modality="csv", **d))
    # standalone images that sit next to a caption sidecar (<name>.txt) are also admitted as image units
    loose = []
    referenced = " ".join(" ".join(d["I_D"]) for d in docs)
    for img in sorted(config.RAW_IMG_DIR.glob("*.*")):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
            continue
        side = img.with_suffix(".txt")
        if side.exists() and img.name not in referenced:
            PARSED_IMG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, PARSED_IMG_DIR / img.name)
            caption = _clean(side.read_text(encoding="utf-8", errors="replace"))
            if caption:
                loose.append((image_url(img.name), caption))
    if loose:
        loose_units = []
        loose_image_records = []
        for index, (url, caption) in enumerate(loose, 1):
            figure_id = f"standalone_images:figure-{index:03d}"
            image_id = f"{figure_id}:panel-01"
            source_name = url.rsplit("/", 1)[-1]
            unit = dict(
                content=caption,
                modality="Image",
                position=index - 1,
                url=url,
                evidence=caption,
                unit_id=image_id,
                image_id=image_id,
                figure_id=figure_id,
                caption_id=f"{figure_id}:caption",
                panel_index=1,
                panel_count=1,
                related_text_ids=[],
                source=source_name,
                source_path=f"data/raw/images/{source_name}",
                doc_id="standalone_images",
                source_modality="image",
                page_number=None,
            )
            caption_unit = dict(
                content=caption,
                modality="Caption",
                position=index - 1,
                evidence=caption,
                unit_id=unit["caption_id"],
                figure_id=figure_id,
                caption_id=unit["caption_id"],
                image_ids=[image_id],
                image_urls=[url],
                image_url=url,
                related_text_ids=[],
                related_figure_ids=[figure_id],
                source=source_name,
                source_path=unit["source_path"],
                doc_id="standalone_images",
                source_modality="image",
                page_number=None,
            )
            loose_units.extend((unit, caption_unit))
            loose_image_records.append({
                "image_id": image_id,
                "figure_id": figure_id,
                "caption_id": unit["caption_id"],
                "panel_index": 1,
                "panel_count": 1,
                "url": url,
                "caption": caption,
                "source": source_name,
                "source_path": unit["source_path"],
                "page_number": None,
                "related_text_ids": [],
                "related_text": [],
            })
        docs.append(dict(doc_id="standalone_images", source="data/raw/images", modality="image",
                         S_D=[], I_D=[u for u, _ in loose], C_I=[c for _, c in loose],
                         units=loose_units, image_records=loose_image_records, parser="sidecar"))
    config.PARSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.PARSED_JSON.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_txt = sum(len(d["S_D"]) for d in docs)
    n_units = sum(len(d.get("units", [])) for d in docs)
    n_img = sum(len(d["I_D"]) for d in docs)
    n_rec = sum(len(d.get("records", [])) for d in docs)
    n_filtered = sum(d.get("filtered_text_fragments", 0) for d in docs)
    ocr_unavailable = [
        (d["source"], page)
        for d in docs
        for page in d.get("ocr_unavailable_pages", [])
    ]
    print(f"parsed {len(docs)} documents -> {config.PARSED_JSON}  "
          f"(parser units={n_units}, text fragments={n_txt}, filtered text={n_filtered}, "
          f"image panels={n_img}, records={n_rec})")
    if ocr_unavailable:
        examples = ", ".join(f"{source} (page {page})" for source, page in ocr_unavailable[:5])
        remaining = len(ocr_unavailable) - 5
        suffix = f", plus {remaining} more" if remaining > 0 else ""
        print(f"[parser] OCR unavailable for {len(ocr_unavailable)} scan-only page(s): "
              f"{examples}{suffix}. Install Tesseract OCR and rerun `python main.py parse`.")
    return docs


if __name__ == "__main__":
    run()
