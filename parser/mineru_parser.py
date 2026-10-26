"""Steps 2-3 - multimodal parsing -> unified JSON.   Paper Sec. II-B, Eq. (1)-(3).

PDF  -> MinerU (text fragments S_D, images I_D, captions C_I)         Eq. (3)
CSV  -> structured records R_si = (t_i, v_i1, ... v_im)                Eq. (1) -> verbalised sentences
Both are merged into data/parsed/documents.json :  D_JSON = {(S_D, I_D, C_I)}

MinerU is used when its CLI (`mineru`, or legacy `magic-pdf`) is on PATH.
Otherwise a PyMuPDF fallback with the same output schema is used (text blocks, image blocks,
"Fig./Figure" caption detection by geometry).
"""
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PARSED_IMG_DIR = config.PARSED_JSON.parent / "images"
CAP_RE = re.compile(r"^\s*(fig\.?|figure|table|图|表)\s*[\dIVX]+", re.I)
_MIN_PROSE_CHARS = 30
_MIN_ALPHA_RATIO = 0.55
_BAD_TEXT_CHARS = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]")


def image_url(filename: str) -> str:
    return f"{config.IMAGE_BASE_URL.rstrip('/')}/{filename}"


def _clean(t: str) -> str:
    if t is None:
        return ""
    text = str(t).replace("\ufeff", "").replace("\ufffd", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_document(doc: dict, source: Path, doc_id: str, modality: str) -> dict:
    """Normalize parser output and ensure every extraction unit has traceable source evidence."""
    try:
        source_path = source.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        source_path = source.name
    normalized = []
    seen = {}
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
        evidence = unit.get("evidence", content)
        if isinstance(evidence, str):
            evidence = _clean(evidence)
        elif isinstance(evidence, dict):
            evidence = {_clean(key): _clean(value) for key, value in evidence.items()}
        unit.update(content=content, source=source.name, source_path=source_path, doc_id=doc_id,
                    source_modality=modality, page_number=page_number, evidence=evidence)
        if "section" in unit:
            unit["section"] = _clean(unit.get("section"))
        normalized.append(unit)
    doc["units"] = normalized
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


def parse_pdf_mineru(pdf: Path, doc_id: str):
    exe = mineru_available()
    out = config.MINERU_OUT_DIR / doc_id
    out.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-p", str(pdf), "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
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
                    fname = f"{doc_id}_{len(I):02d}{src.suffix}"
                    shutil.copy(src, PARSED_IMG_DIR / fname)
                    I.append(image_url(fname))
                    C.append(cap)
                    units.append(dict(content=cap, modality="Table", position=it.get("page_idx", 0),
                                      url=I[-1], table_caption=cap))
                    table_image_added = True
            if cap and not table_image_added and not table_records:
                units.append(dict(content=cap, modality="Table", position=it.get("page_idx", 0),
                                  table_caption=cap))
        elif typ == "image" and it.get("img_path"):
            src = cl.parent / it["img_path"]
            if not src.exists():
                continue
            fname = f"{doc_id}_{len(I):02d}{src.suffix}"
            shutil.copy(src, PARSED_IMG_DIR / fname)
            caps = it.get("img_caption") or it.get("image_caption") or []
            caps = caps if isinstance(caps, list) else [caps]
            cap = _clean(" ".join(str(item) for item in caps if item)) if caps else ""
            I.append(image_url(fname))
            C.append(cap)
            units.append(dict(content=cap, modality="Table" if typ == "table" else "Image",
                              position=it.get("page_idx", 0), url=I[-1]))
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

    for pno, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        texts = []                                          # (bbox, text, size, bold)
        for b in blocks:
            if b["type"] != 0:
                continue
            spans = [sp for l in b["lines"] for sp in l["spans"] if sp["text"].strip()]
            t = _clean(" ".join(sp["text"] for sp in spans))
            if t:
                bold = all((sp["flags"] & 16) or "bold" in sp["font"].lower() for sp in spans)
                texts.append((b["bbox"], t, max(sp["size"] for sp in spans), bold))
        used = set()

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
                fname = f"{doc_id}_{len(I):02d}.png"
                page.get_pixmap(clip=(bb + (-4, -4, 4, 4)) & page.rect, dpi=150).save(str(PARSED_IMG_DIR / fname))
                I.append(image_url(fname))
                C.append(cap)
                units.append(dict(content=cap, modality="Table", position=pno, url=I[-1], section=cur_section))
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
            key = hash(b["image"])
            if key in seen_img:
                continue
            seen_img.add(key)
            below = [(i, x) for i, x in enumerate(texts) if x[0][1] >= b["bbox"][3] - 5 and i not in used]
            below.sort(key=lambda x: x[1][0][1])
            cap = ""
            pick = next((x for x in below[:2] if FIG_CAP_RE.match(x[1][1])), None)
            if pick:
                cap = pick[1][1]
                used.add(pick[0])
            if not cap and config.IMAGE_REQUIRE_CAPTION:
                continue
            fname = f"{doc_id}_{len(I):02d}.{b.get('ext', 'png')}"
            (PARSED_IMG_DIR / fname).write_bytes(b["image"])
            I.append(image_url(fname))
            C.append(cap)
            units.append(dict(content=cap, modality="Image", position=pno, url=I[-1], section=cur_section))

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
            cl = pymupdf.Rect(x0 - 30, y0 - 22, x1 + 10, bb[1] - 2) & page.rect     # include axis titles / tick labels
            fname = f"{doc_id}_{len(I):02d}.png"
            page.get_pixmap(clip=cl, dpi=150).save(str(PARSED_IMG_DIR / fname))
            I.append(image_url(fname))
            C.append(t)
            used.add(i)
            units.append(dict(content=t, modality="Image", position=pno, url=I[-1], section=cur_section))
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
                units.append(dict(content=t, modality="Formula", position=pno, section=cur_section))
            else:
                reason = _pdf_text_rejection(t)
                if reason:
                    filtered_text += 1
                    continue
                units.append(dict(content=t, modality="Text", position=pno, section=cur_section))
            S.append(t)
    for sec in sections:
        sec.pop("size", None), sec.pop("y1", None)
    return dict(S_D=S, I_D=I, C_I=C, units=units, records=records, formulas=formulas, sections=sections,
                parser="pymupdf-fallback", filtered_text_fragments=filtered_text)


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
            print(f"[parser] MinerU failed on {pdf.name} ({type(e).__name__}: {str(e)[:100]}); using PyMuPDF fallback")
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
        docs.append(dict(doc_id="standalone_images", source="data/raw/images", modality="image",
                         S_D=[], I_D=[u for u, _ in loose], C_I=[c for _, c in loose],
                         units=[dict(content=c, modality="Image", position=i, url=u, evidence=c,
                                     source=img.name, source_path=f"data/raw/images/{img.name}",
                                     doc_id="standalone_images", source_modality="image", page_number=None)
                                for i, (u, c) in enumerate(loose)],
                         parser="sidecar"))
    config.PARSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.PARSED_JSON.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_txt = sum(len(d["S_D"]) for d in docs)
    n_img = sum(len(d["I_D"]) for d in docs)
    n_rec = sum(len(d.get("records", [])) for d in docs)
    n_filtered = sum(d.get("filtered_text_fragments", 0) for d in docs)
    print(f"parsed {len(docs)} documents -> {config.PARSED_JSON}  "
          f"(text fragments={n_txt}, filtered text={n_filtered}, images={n_img}, records={n_rec})")
    return docs


if __name__ == "__main__":
    run()
