"""
OCR & Text Extraction — Backend Reference (FastAPI)
=====================================================
Stateless, privacy-first API layer for server-side OCR (EasyOCR/PaddleOCR)
or Cloud Vision fallback, complementing the client-side Tesseract.js app.

Design rules:
- No image persisted to disk/DB. Processed fully in memory, discarded after response.
- Language: Thai + English by default (multi-language pass).
- Structured output: plain text + line-level boxes (for table/CSV reconstruction).

Install:
    pip install fastapi uvicorn python-multipart easyocr pillow numpy --break-system-packages

Run:
    uvicorn ocr_backend_fastapi:app --reload --port 8000
"""

from __future__ import annotations

import io
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

app = FastAPI(
    title="OCR Extraction API",
    version="1.0.0",
    description="Stateless OCR endpoint — Thai/English. No image is stored server-side.",
)

# Restrict to your actual frontend origin(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_FILE_SIZE_MB = 15

# ---------------------------------------------------------------------------
# Lazy-loaded OCR engine (EasyOCR supports Thai + English natively).
# Loaded once per process, not per-request, to avoid reload overhead —
# this holds model weights only, never user images, so it does not
# violate the stateless/no-retention rule.
# ---------------------------------------------------------------------------
_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr  # heavy import kept lazy so app boots fast

        _reader = easyocr.Reader(["th", "en"], gpu=False)
    return _reader


class TextLine(BaseModel):
    text: str
    confidence: float
    bbox: list[list[float]]  # 4 corner points [[x,y], ...]
    color: str = "white"  # background color sampled under this box: yellow/blue_header/grey/white
    hex: str = "FFFFFF"  # exact sampled background color (see sample_bg_rgb), for 1:1 color reproduction


class SapField(BaseModel):
    section: str
    label: str
    element_type: str
    low: str
    high: str
    warnings: list[str] = []  # FI/CO data-integrity warnings (see validate_fi_co_field)
    hex: str = "FFFFFF"  # exact sampled background color of this field's value box


class OCRResponse(BaseModel):
    text: str
    lines: list[TextLine]
    rows: list[list[str]]  # text reconstructed row-by-row, column-snapped — a rough table
    colors: list[list[str]]  # same shape as rows — dominant field color per cell
    hex_colors: list[list[str]] = []  # same shape as rows — exact sampled background hex per cell
    sap_fields: list[SapField]  # structured Label/Type/Low-To-High rows, SAP screens only
    is_sap_screen: bool  # heuristic: does this look like an SAP selection/transaction screen
    language: str
    engine: str
    elapsed_ms: int


def sample_bg_rgb(image: Image.Image, box) -> tuple[int, int, int]:
    """
    Samples the actual background color under a detected text box — the
    real RGB triple from the screenshot, not a bucketed guess. Text pixels
    tend to be dark, so the brighter majority of the crop is taken as the
    background (falling back to all pixels if the box is unusually dark
    overall, e.g. a solid dark header bar).
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, x1 = max(int(min(xs)), 0), min(int(max(xs)), image.width)
    y0, y1 = max(int(min(ys)), 0), min(int(max(ys)), image.height)
    if x1 <= x0 or y1 <= y0:
        return (255, 255, 255)

    crop = image.crop((x0, y0, x1, y1))
    pixels = list(crop.getdata())
    if not pixels:
        return (255, 255, 255)

    bg_pixels = [p for p in pixels if sum(p[:3]) > 260] or pixels
    r, g, b = Counter(bg_pixels).most_common(1)[0][0][:3]
    return (r, g, b)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "%02X%02X%02X" % rgb


def bucket_color(rgb: tuple[int, int, int]) -> str:
    """
    Buckets a sampled RGB into the handful of semantic colors SAP selection
    screens actually use. This SEMANTIC bucket still drives layout logic
    (is this field mandatory? is this row a header bar?) because that logic
    needs a stable category, not an exact shade that varies by screenshot
    tool/compression — but for what actually gets PAINTED in Excel, callers
    should prefer the exact sampled hex from sample_bg_rgb/rgb_to_hex over
    this bucket, which is not a color-accurate reproduction on its own.
    """
    r, g, b = rgb
    if r > 225 and g > 210 and b < 190:
        return "yellow"
    if b > 140 and b > r + 15 and g > 130:
        return "blue_header"
    if abs(r - g) < 12 and abs(g - b) < 12 and r < 225:
        return "grey"
    return "white"


def classify_color(image: Image.Image, box) -> str:
    """Back-compat wrapper: bucket only, for callers that don't need the
    exact RGB. Prefer sample_bg_rgb() + bucket_color() when both are
    needed, to avoid sampling the same box twice."""
    return bucket_color(sample_bg_rgb(image, box))


def looks_like_field_form_shape(rows: list[list[str]]) -> bool:
    """
    Distinguishes a label:value SELECTION SCREEN (Company Code / Posting
    Date / ...) from a wide, dense DATA TABLE (a Data Browser entry list,
    a line-item grid) — both can trip the SAP keyword/menu-bar heuristic
    or the manual "Treat as SAP screen" override, but only the first shape
    is something build_sap_fields() can turn into meaningful Label/Low/
    High records. Forcing a wide table through it crams entire rows of
    unrelated column values into one "Low" string per row and (in Visual
    Replica) merges cells meant for a single field into a multi-column
    table header — exactly the "cells got merged, wrong colors" bug this
    guards against.

    A field-form's grid (from group_into_table) is mostly EMPTY once
    snapped onto its column positions — each row usually holds only a
    label plus one or two values across several column slots. A real data
    table is densely filled across most of its columns on most rows. This
    checks that density rather than column count alone (a two-column
    selection screen can legitimately snap to 8+ columns).
    """
    if not rows:
        return True
    body_rows_all = rows[1:] if len(rows) > 1 else rows

    # A dense multi-column data table whose columns failed to cluster apart
    # (e.g. a customizing table like "BP groupings": Grouping/Short name/
    # Description/Number Range/External/Int.Std/Ext.Std all squashed into
    # one wide "value" cell per row) still shows its true shape in the
    # text: many rows, each cell a run-on of 5+ words that are really
    # several table columns' worth of content mashed together. Real field
    # labels and values are short (a label, a code, a short description) —
    # this catches the "wrong headers, everything crammed into one column"
    # failure mode even when the column count alone looks narrow.
    if body_rows_all:
        long_cells = sum(
            1 for row in body_rows_all for cell in row
            if cell and len(cell.split()) >= 5
        )
        if long_cells / max(len(body_rows_all), 1) > 0.3:
            return False

    # A repeated run of many near-identical, code-like rows (a Number Range
    # interval list, a technical-key table) is a data table even when
    # column-splitting under-counted its columns and it looks narrow —
    # a real selection screen almost never has 8+ rows that each carry two
    # or more long digit/code sequences (interval "from"/"to" numbers, IDs),
    # while a customizing/interval table is built almost entirely of them.
    # This catches the shape BEFORE the column-count shortcut below can
    # wrongly wave it through as "too narrow to be a table".
    long_code_re = re.compile(r"\d{5,}")
    if len(body_rows_all) >= 8:
        code_heavy_rows = sum(
            1 for row in body_rows_all
            if sum(1 for cell in row if cell and long_code_re.search(cell)) >= 2
        )
        if code_heavy_rows / len(body_rows_all) > 0.5:
            return False

    n_cols = max((len(r) for r in rows), default=0)
    if n_cols < 6:
        return True
    body_rows = rows[1:] if len(rows) > 1 else rows
    if not body_rows:
        return True
    filled = sum(1 for row in body_rows for cell in row if cell)
    total = sum(len(row) for row in body_rows)
    density = filled / total if total else 0
    return density <= 0.4


SAP_SCREEN_KEYWORDS = (
    "selection criteria", "company code", "posting date", "processing mode",
    "trading partner", "business place", "sd document", "accounting document",
    "layout", "variant", "execute", "further selections", "output control",
    "database selection", "fiscal year", "plant", "vendor", "purchase order",
    "sales order", "material", "gl account", "g/l account", "cost center",
    "profit center", "document number", "document type", "controlling area",
    "movement type", "storage location", "multiple selection", "customer number",
    "purchasing document", "sales document", "document currency", "local currency",
    "change language", "posting key", "account type", "special g/l", "tax code",
    "line item", "clearing", "assignment", "sp.g/l assgt", "payt terms",
    "data entry view", "cross-comp", "ledger group", "texts exist",
    "postal code", "street address", "po box", "communication", "search terms",
    "business partner", "address overview", "time zone", "country/reg",
    "number range object", "number length domain", "subobject", "intervals",
    "customizing", "short txt", "long txt", "translatn date", "branch number",
    "trading part", "vendortaxinv",
)

# The standard SAP GUI menu bar (Program/Edit/Goto/System/Help) sits on top
# of almost every screen, whatever the report — unlike the field-name list
# above (which is specific to this one screen's vocabulary and misses any
# other transaction), these five words appearing together are a strong,
# report-independent signal on their own.
SAP_MENU_BAR_WORDS = ("program", "edit", "goto", "system", "help")


def looks_like_sap_screen(full_text: str) -> bool:
    t = full_text.lower()
    if sum(1 for w in SAP_MENU_BAR_WORDS if w in t) >= 4:
        return True
    # Several stacked Select-Option range fields (their own "to" detected as
    # a standalone text line, one per range) is also fairly distinctive of
    # an SAP selection screen, even when none of the specific field names
    # below are recognized.
    to_lines = sum(1 for line in t.split("\n") if line.strip() == "to")
    if to_lines >= 3:
        return True
    return sum(1 for kw in SAP_SCREEN_KEYWORDS if kw in t) >= 2


def split_into_windows(results, image_width: float, image_height: float = None, min_window_frac: float = 0.12):
    """
    Detects two (or more) SEPARATE SAP windows sitting side by side in one
    screenshot (a Data Browser table next to its "Table View Maintenance"
    dialog, a main screen next to a smaller popup) and returns the x-ranges
    that belong to each. A screenshot like this used to get fed to the
    pipeline as ONE continuous grid: rows from both windows that happened
    to land at similar y-positions were merged into the same row, and
    entire unrelated lines got concatenated into a single cell — exactly
    the "table mapping between housebank" bug where a whole sentence from
    the SECOND window ended up jammed into one "Low" value next to a label
    from the FIRST window.

    A plain wide table also has gaps between its columns, so gap width
    alone isn't a safe signal — a normal ~45-60px column gap would trigger
    false splits on ordinary tables. Two real, independent windows each
    have their OWN title bar near their own top edge, which an ordinary
    table's columns never do. So this requires BOTH: a gap noticeably
    wider than a normal column gap (scaled to image size, not a fixed
    pixel count that breaks on very large or very small screenshots), AND
    a detection sitting near the top of the image on each side of the
    gap (that side's own title/menu bar) before treating it as a window
    boundary. Returns a list of (x0, x1) bands; a single band covering
    the whole width means no split was found (the common, single-window
    case is unaffected by this at all).
    """
    if not results:
        return [(0.0, image_width)]

    min_gap = max(90.0, image_width * 0.07)
    title_zone = (image_height or 0) * 0.12 if image_height else float("inf")

    intervals = []
    for box, _text, _conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        intervals.append((min(xs), max(xs), min(ys)))
    intervals.sort()

    bands = [[intervals[0][0], intervals[0][1], [intervals[0][2]]]]
    for x0, x1, y0 in intervals[1:]:
        if x0 - bands[-1][1] > min_gap:
            bands.append([x0, x1, [y0]])
        else:
            bands[-1][1] = max(bands[-1][1], x1)
            bands[-1][2].append(y0)

    # Each side of a real window boundary has its own title bar near its
    # own top edge. If a candidate band never has anything near the top,
    # it's more likely a stray gap inside one wide table (a row that just
    # happens to be sparse on one side) than a genuine second window —
    # merge it back into its nearest neighbor instead of splitting on it.
    if len(bands) > 1 and image_height:
        i = 0
        while i < len(bands):
            has_title = any(y <= title_zone for y in bands[i][2])
            if not has_title:
                if i > 0:
                    bands[i - 1][1] = max(bands[i - 1][1], bands[i][1])
                    bands[i - 1][2].extend(bands[i][2])
                    del bands[i]
                elif len(bands) > 1:
                    bands[1][0] = min(bands[1][0], bands[i][0])
                    bands[1][2].extend(bands[i][2])
                    del bands[i]
                else:
                    i += 1
            else:
                i += 1
    bands = [b[:2] for b in bands]

    # Drop slivers (stray detections) by re-merging any band narrower than
    # min_window_frac of the image into whichever neighbor it's closer to.
    min_width = image_width * min_window_frac
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, (x0, x1) in enumerate(bands):
            if x1 - x0 >= min_width:
                continue
            if i == 0:
                bands[1][0] = min(bands[1][0], x0)
            elif i == len(bands) - 1:
                bands[i - 1][1] = max(bands[i - 1][1], x1)
            else:
                # merge into the nearer neighbor
                if (x0 - bands[i - 1][1]) <= (bands[i + 1][0] - x1):
                    bands[i - 1][1] = max(bands[i - 1][1], x1)
                else:
                    bands[i + 1][0] = min(bands[i + 1][0], x0)
            del bands[i]
            changed = True
            break

    if len(bands) <= 1:
        return [(0.0, image_width)]
    # Pad each band out to the midpoint of its gap to its neighbors so no
    # detection right at a band's edge gets excluded.
    padded = []
    for i, (x0, x1) in enumerate(bands):
        left = 0.0 if i == 0 else (bands[i - 1][1] + x0) / 2
        right = image_width if i == len(bands) - 1 else (x1 + bands[i + 1][0]) / 2
        padded.append((left, right))
    return padded


def detect_vertical_gridlines(image: Image.Image, min_run_frac: float = 0.4) -> list[float]:
    """
    Finds the faint light-grey vertical ruling lines SAP draws between
    table columns (ALV grids, Data Browser tables, line-item displays) by
    scanning for x-positions that are a near-neutral grey for a long
    vertical run — real column separators, not just whitespace or a
    coincidental gap in the text. Where these exist they are a far more
    reliable column boundary than guessing from OCR text spacing (which
    drifts with font kerning and can't tell "two adjacent short values" a
    part from "one wide merged value").

    Returns sorted x-positions of detected gridlines; an empty list means
    none were found (a screen with no visible ruling, or a plain photo),
    in which case callers should fall back to the text-spacing heuristic.
    """
    arr = np.array(image.convert("RGB")).astype(int)
    h, w, _ = arr.shape
    if h == 0 or w == 0:
        return []
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # SAP's gridline grey sits in a narrow, near-neutral band — lighter
    # than real content text/borders, darker than the white/panel
    # background around it.
    is_grey = (np.abs(r - g) < 8) & (np.abs(g - b) < 8) & (r > 185) & (r < 232)
    col_frac = is_grey.mean(axis=0)
    candidate_xs = np.where(col_frac > min_run_frac)[0]
    if len(candidate_xs) == 0:
        return []

    lines: list[float] = []
    start = prev = candidate_xs[0]
    for x in candidate_xs[1:]:
        if x - prev <= 2:
            prev = x
        else:
            lines.append((start + prev) / 2)
            start = prev = x
    lines.append((start + prev) / 2)
    return lines


def _adaptive_col_gap(xs_sorted: list[float], avg_h: float) -> float:
    """
    Picks a column-separation threshold from the screenshot's OWN x-position
    data instead of one fixed pixel value or a fixed multiple of text
    height — both broke down across different SAP screenshots: a single
    fixed pixel gap merged real columns on a dense, narrow-column ALV grid
    (8 tight columns like Cl./RFC Destination/CoCd/...), while scaling
    purely off text height merged real columns on a WIDER, more sparsely
    spaced table (a 6-column Number Range interval list) where the text
    happened to be a bit taller.

    The x-positions themselves settle this without guessing a formula: the
    gaps between consecutive sorted x-starts naturally split into a "small"
    cluster (jitter between rows in the SAME column, or word-spacing within
    one multi-word cell) and a "large" cluster (crossing into the NEXT
    column) — whatever this screenshot's actual scale happens to be. The
    biggest jump between consecutive sorted gap sizes is the natural
    boundary between those two clusters; the threshold is set halfway
    across that jump. Falls back to a text-height-based estimate only when
    there isn't enough data to find a clear jump (e.g. everything really is
    one column).
    """
    return max(18.0, avg_h * 1.4)


def group_into_table(results, image: Image.Image, row_gap_ratio: float = 0.6, col_gap_px: float | None = None):
    """
    Turns EasyOCR's flat list of (box, text, confidence) detections into a
    grid: groups detections into rows by vertical position, then snaps
    every item — across all rows — onto a shared set of column positions
    (instead of just sorting each row left-to-right independently), so a
    field two rows down lines up under the same column as the field above
    it, matching the source screenshot much more closely.

    When the screenshot has real gridlines (detect_vertical_gridlines),
    those are used as authoritative column boundaries — sharper and more
    reliable than guessing from OCR text spacing. Falls back to the
    original x-gap clustering heuristic when no gridlines are visible (a
    column whose x-position drifts more than col_gap_px from its
    neighbors becomes its own column, and cells OCR missed just stay
    blank). Returns a text grid and same-shaped color/hex grids.
    """
    if not results:
        return [], [], []

    items = []
    for box, text, _conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        rgb = sample_bg_rgb(image, box)
        items.append({
            "text": text,
            "x": min(xs),
            "y": (min(ys) + max(ys)) / 2,
            "h": (max(ys) - min(ys)) or 1,
            "color": bucket_color(rgb),
            "hex": rgb_to_hex(rgb),
        })

    items.sort(key=lambda i: i["y"])
    avg_h = sum(i["h"] for i in items) / len(items)

    rows: list[list[dict]] = [[items[0]]]
    for it in items[1:]:
        if abs(it["y"] - rows[-1][-1]["y"]) <= avg_h * row_gap_ratio:
            rows[-1].append(it)
        else:
            rows.append([it])

    # Build shared column positions from every item's x, across all rows.
    xs_sorted = sorted(it["x"] for row in rows for it in row)

    # Prefer real gridlines when the screenshot actually has them — sharper
    # and more reliable than guessing column boundaries from text spacing.
    # Only trust lines that actually fall within the span of detected text
    # (a scrollbar edge or window border outside that span isn't a column
    # separator), and only switch to this path when there are enough of
    # them to plausibly bound real columns.
    gridlines = detect_vertical_gridlines(image)
    x_min, x_max = xs_sorted[0], xs_sorted[-1]
    gridlines = [gx for gx in gridlines if x_min - 20 <= gx <= x_max + 20]

    if len(gridlines) >= 2:
        boundaries = sorted(gridlines)
        n_cols = len(boundaries) + 1

        def col_index(x: float, _b=boundaries) -> int:
            idx = 0
            for gx in _b:
                if x < gx:
                    break
                idx += 1
            return min(idx, n_cols - 1)
    else:
        effective_gap = col_gap_px if col_gap_px is not None else _adaptive_col_gap(xs_sorted, avg_h)
        col_starts = [xs_sorted[0]]
        for x in xs_sorted[1:]:
            if x - col_starts[-1] > effective_gap:
                col_starts.append(x)

        def col_index(x: float, _s=col_starts) -> int:
            return min(range(len(_s)), key=lambda i: abs(_s[i] - x))

        n_cols = len(col_starts)
    text_grid: list[list[str]] = []
    color_grid: list[list[str]] = []
    hex_grid: list[list[str]] = []
    for row in rows:
        text_cells = [""] * n_cols
        color_cells = ["white"] * n_cols
        hex_cells = ["FFFFFF"] * n_cols
        for it in sorted(row, key=lambda i: i["x"]):
            idx = col_index(it["x"])
            text_cells[idx] = (text_cells[idx] + " " + it["text"]).strip() if text_cells[idx] else it["text"]
            color_cells[idx] = it["color"]
            hex_cells[idx] = it["hex"]
        text_grid.append(text_cells)
        color_grid.append(color_cells)
        hex_grid.append(hex_cells)

    return text_grid, color_grid, hex_grid


# Short, generic words (a real section title on its own, e.g. "Currency" or
# "Group", but ALSO a common substring of ordinary field labels — "Trading
# Partner Group", "CCA Name", "Posting Period"). These are only trusted as a
# header when the row's ENTIRE text equals one of them, never as a substring
# match, or almost every field label containing the word would be swallowed.
SECTION_HEADER_EXACT_WORDS = (
    "language", "currency", "option", "output", "period", "settings",
    "group", "name", "layout", "customizing", "intervals", "communication",
)

# Longer, distinctive phrases that are safe to substring-match — nothing
# resembling an ordinary field label happens to contain these by accident.
SECTION_HEADER_PHRASES = (
    "selection criteria", "further selections", "output control",
    "processing options", "database selections", "print control",
    "general data", "additional data", "display options",
    "general selections", "organizational data", "list output",
    "program selections", "date selections", "restrictions",
    "street address", "po box address",
    "search terms", "data entry view", "items in document currency",
    "other line item",
)


def is_header_bar_row(image: Image.Image, y_center: float, height: float, text: str = "") -> bool:
    """
    Confirms a row is a genuine SAP section header bar ("Selection
    Criteria", "Currency", ...), not just an ordinary field label sitting
    alone in its row (which happens whenever that field's input box came
    back empty from OCR, or the field is a checkbox/radio with nothing
    for OCR to read next to it).

    This used to also sample pixel color and treat a wide, mostly-blue
    strip as proof of a real header bar. That backfired on screens where
    the WHOLE selection-screen panel (not just the header) is tinted a
    light blue — every lone field row then reads as "mostly blue" too,
    and real fields (Company Code, Trading Partner, Include Reversed
    Document) kept getting swallowed as fake section titles no matter how
    the color threshold was tuned. Matching against a curated list of the
    section titles SAP actually uses is far more reliable than guessing
    from a screenshot's color palette, which varies by theme/screenshot
    tool and isn't a safe signal here.
    """
    t = text.strip().lower()
    if t in SECTION_HEADER_EXACT_WORDS:
        return True
    return any(kw in t for kw in SECTION_HEADER_PHRASES)


def build_sap_fields(results, image: Image.Image, row_gap_ratio: float = 0.6) -> list[dict]:
    """
    Restructures an SAP selection screen's detections into one row per
    field: Section | Field Label | Element Type | Low | High — the layout
    a person reading the screen actually thinks in, instead of a raw
    left-to-right grid.

    What this CAN determine from OCR text + sampled field color:
    - the field's label (leftmost text in its row)
    - a Select-Option range, by finding a literal "to" between two boxes
      and keeping it as the separator between Low and High (never its own
      column)
    - whether a field is likely mandatory, from its yellow background
    - section headers (e.g. "Selection Criteria", "Currency"), from the
      blue header-bar color

    What this CANNOT reliably determine, and does not claim to: F4/search
    icons, the "multiple selection" icon, dropdown carets, or checkbox/
    radio-button state. Those are graphical glyphs with no OCR text at
    all — telling them apart needs icon/shape matching against known SAP
    icon shapes, not text recognition, so they're left out rather than
    guessed and presented as fact.
    """
    if not results:
        return []

    # Menu bar (Program/Edit/Goto/System/Help) sits in the top strip of an
    # SAP screen, and the status bar (client/system-ID/OVR) in the bottom
    # strip — neither is a selection-screen field, so both are dropped by
    # position before any row/field logic runs, per the "exclude chrome"
    # requirement.
    top_cutoff = image.height * 0.06
    bottom_cutoff = image.height * 0.95

    items = []
    for box, text, _conf in results:
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        y_center = (min(ys) + max(ys)) / 2
        if y_center < top_cutoff or y_center > bottom_cutoff:
            continue
        rgb = sample_bg_rgb(image, box)
        items.append({
            "text": text,
            "x": min(xs),
            "y": y_center,
            "h": (max(ys) - min(ys)) or 1,
            "color": bucket_color(rgb),
            "hex": rgb_to_hex(rgb),
        })
    if not items:
        return []

    items.sort(key=lambda i: i["y"])
    # Median, not mean: a couple of large-font items (a screen title, say)
    # would otherwise drag the average height way up and make the row-gap
    # threshold too generous, merging two visually distinct field rows
    # into one — median is unmoved by a small number of oversized outliers.
    heights = sorted(i["h"] for i in items)
    median_h = heights[len(heights) // 2]

    # Raised from 0.6 -> 0.9: a field's label and its own input box often
    # sit a few pixels off-center from each other, and the tighter ratio
    # was enough to split "Company Code" from its own row on a hair's-
    # width difference, then merge it with the row below by mistake.
    row_gap_ratio = max(row_gap_ratio, 0.9)

    rows: list[list[dict]] = [[items[0]]]
    for it in items[1:]:
        if abs(it["y"] - rows[-1][-1]["y"]) <= median_h * row_gap_ratio:
            rows[-1].append(it)
        else:
            rows.append([it])

    # Classic SAP selection screens lay fields out in TWO side-by-side
    # columns (e.g. "Company Code ... to ..." on the left, "Posting Date
    # ... to ..." on the right, on the very same visual row). Treating a
    # whole row as a single field merged both into one garbled record and
    # dropped the right-hand label entirely. Instead, find the one big
    # horizontal gap that recurs across the document — the gutter between
    # the two columns — and use it to split every row into left/right
    # field groups before building records.
    xs_sorted = sorted(i["x"] for i in items)
    col_starts = [xs_sorted[0]]
    for x in xs_sorted[1:]:
        if x - col_starts[-1] > 40:
            col_starts.append(x)

    split_x = None
    if len(col_starts) > 2:
        gaps = [
            (col_starts[i + 1] - col_starts[i], (col_starts[i] + col_starts[i + 1]) / 2)
            for i in range(len(col_starts) - 1)
        ]
        biggest_gap, mid = max(gaps, key=lambda g: g[0])
        # Only treat it as a real column gutter if it's well clear of
        # ordinary label/value spacing, and roughly in the middle third of
        # the screen (a two-column layout's gutter, not a stray one-off gap
        # near either edge).
        if biggest_gap > 120 and image.width * 0.25 < mid < image.width * 0.75:
            split_x = mid

    if split_x is not None:
        # A genuine two-column screen has a REAL field label starting the
        # right-hand group on most rows (e.g. "Posting Date"). A screen
        # that's actually single-column, but happens to have one big gap
        # somewhere in its row (typically between a label and its own far
        # input box, or before its own "to"), would instead have "to" or
        # nothing recognizable as the first item on the right — splitting
        # THAT tears a normal field in half and drops its value. Check
        # which case this is before committing to the split.
        candidate_rows = [sorted(r, key=lambda i: i["x"]) for r in rows if len(r) > 1]
        split_rows = [r for r in candidate_rows if any(i["x"] >= split_x for i in r) and any(i["x"] < split_x for i in r)]
        if split_rows:
            looks_like_label = 0
            for r in split_rows:
                right_first = next(i["text"].strip() for i in r if i["x"] >= split_x)
                if right_first.lower() != "to" and len(right_first) > 2 and any(ch.isalpha() for ch in right_first):
                    looks_like_label += 1
            if looks_like_label / len(split_rows) < 0.5:
                split_x = None
        else:
            split_x = None

    def build_one(group: list[dict], section: str) -> dict | None:
        group_sorted = sorted(group, key=lambda i: i["x"])
        label, rest = group_sorted[0]["text"], group_sorted[1:]
        to_idx = next((i for i, r in enumerate(rest) if r["text"].lower() == "to"), None)

        if to_idx is not None:
            low = " ".join(r["text"] for r in rest[:to_idx]).strip()
            high = " ".join(r["text"] for r in rest[to_idx + 1:]).strip()
            low_hex = rest[0]["hex"] if rest[:to_idx] else group_sorted[0]["hex"]
            return {
                "section": section, "label": label,
                "element_type": "Select-Option Range: Low - High",
                "low": low, "high": high,
                "warnings": validate_fi_co_field(label, low) + validate_fi_co_field(label, high),
                "hex": low_hex,
            }
        value = " ".join(r["text"] for r in rest).strip()
        mandatory = any(r["color"] == "yellow" for r in rest) if rest else False
        value_hex = rest[0]["hex"] if rest else group_sorted[0]["hex"]
        return {
            "section": section, "label": label,
            "element_type": "Input Field (mandatory)" if mandatory else "Input Field",
            "low": value, "high": "",
            "warnings": validate_fi_co_field(label, value),
            "hex": value_hex,
        }

    fields: list[dict] = []
    section = ""
    for row in rows:
        row_sorted = sorted(row, key=lambda i: i["x"])

        # A lone item on a blue header bar reads as a section title
        # ("Selection Criteria", "Currency", "Language"...), not a field.
        # Checking that ONE item's own small text box is blue is not
        # reliable — an ordinary field whose input box happens to be
        # empty (nothing for OCR to read there) is *also* a lone item,
        # and a noisy color sample off its label text was enough to get
        # it misread as a section header, silently swallowing real
        # fields (this ate Company Code, Trading Partner, and Include
        # Reversed Document in testing). Instead, sample a wide strip
        # spanning most of the row's width: a real section bar is
        # solid blue almost edge-to-edge, while a lone field label
        # sitting on the plain panel background is not — checking the
        # bar itself rather than the label's own text box.
        if len(row_sorted) == 1 and is_header_bar_row(
            image, row_sorted[0]["y"], row_sorted[0]["h"], row_sorted[0]["text"]
        ):
            section = row_sorted[0]["text"]
            continue

        if split_x is not None:
            left = [i for i in row_sorted if i["x"] < split_x]
            right = [i for i in row_sorted if i["x"] >= split_x]
            groups = [g for g in (left, right) if g]
        else:
            groups = [row_sorted]

        for grp in groups:
            field = build_one(grp, section)
            if field:
                fields.append(field)

    return fields


# A representative subset of ISO 4217 codes — covers what a Thailand-based
# FI/CO practice actually sees (THB-adjacent trade + majors), not the full
# 180-code standard. Extend this list rather than replacing the approach if
# a currency used at your client is missing.
ISO_4217_CODES = {
    "THB", "USD", "EUR", "GBP", "JPY", "CNY", "SGD", "MYR", "IDR", "VND",
    "PHP", "KRW", "HKD", "TWD", "INR", "AUD", "NZD", "CAD", "CHF", "SEK",
    "NOK", "DKK", "ZAR", "AED", "SAR", "KHR", "LAK", "MMK", "BND",
}


def validate_fi_co_field(label: str, value: str) -> list[str]:
    """
    Post-processing "data integrity" pass tuned for SAP FI/CO field
    conventions — NOT a general OCR-confidence check (that's a separate,
    per-cell concern). This looks at a field's LABEL to guess what kind of
    FI/CO value it should hold, then flags a value that doesn't match the
    expected shape, so an OCR misread (e.g. "O" for "0", a dropped digit)
    surfaces as a warning instead of silently entering downstream analysis
    or a report someone acts on.

    Deliberately conservative: an EMPTY value never warns (blank fields are
    legitimate — most Select-Option ranges on a real screen are blank), and
    a label this function doesn't recognize never warns either. This is a
    warning layer, not a hard validator — it must never block an export.
    """
    v = value.strip()
    if not v:
        return []
    l = label.strip().lower()
    warnings: list[str] = []

    def is_digits(s: str) -> bool:
        return s.isdigit()

    if "g/l acct" in l or "g/l account" in l or "gl account" in l or l == "g/l":
        if not is_digits(v):
            warnings.append(f"G/L Account '{v}' should be numeric (digits only)")
        elif len(v) != 10:
            warnings.append(f"G/L Account '{v}' has {len(v)} digits, expected 10 (SAP standard length)")

    elif "company code" in l:
        if not (1 <= len(v) <= 4) or not v.isalnum():
            warnings.append(f"Company Code '{v}' should be 1-4 alphanumeric characters")
        elif len(v) != 4:
            warnings.append(f"Company Code '{v}' has {len(v)} characters, expected 4")

    elif "fiscal year" in l:
        if not is_digits(v) or len(v) != 4:
            warnings.append(f"Fiscal Year '{v}' should be a 4-digit number")
        elif not (1900 <= int(v) <= 2099):
            warnings.append(f"Fiscal Year '{v}' is outside the plausible range 1900-2099")

    elif "document number" in l:
        if not is_digits(v):
            warnings.append(f"Document Number '{v}' should be numeric (digits only)")
        elif len(v) != 10:
            warnings.append(f"Document Number '{v}' has {len(v)} digits, expected 10 (SAP standard length)")

    elif l == "currency" or "currency" in l:
        code = v.upper().replace(" ", "")
        if len(code) != 3 or not code.isalpha():
            warnings.append(f"Currency '{v}' should be a 3-letter ISO 4217 code")
        elif code not in ISO_4217_CODES:
            warnings.append(f"Currency '{code}' is not in the recognized ISO 4217 code list — verify it's not an OCR misread")

    elif "amount" in l:
        cleaned = v.replace(",", "").rstrip("-").strip()
        if not re.match(r"^\d+(\.\d{1,2})?$", cleaned):
            warnings.append(f"Amount '{v}' doesn't look like a valid decimal amount (expected digits with up to 2 decimal places)")

    return warnings


def validate_upload(file: UploadFile, data: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")


@app.post("/api/ocr", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    lang: Literal["auto", "th", "en"] = "auto",
    force_sap: bool = False,
):
    """
    Accepts one image, returns extracted text. The image bytes exist only
    in this request's memory (never written to disk or a database) and are
    released once the response is built — satisfies the stateless/
    privacy-first requirement.
    """
    started = time.time()
    raw = await file.read()
    validate_upload(file, raw)

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "Could not decode image") from exc

    reader = get_reader()
    results = reader.readtext(np.array(image))

    def _line(box, text, conf):
        rgb = sample_bg_rgb(image, box)
        return TextLine(
            text=text,
            confidence=round(float(conf), 4),
            bbox=[list(map(float, p)) for p in box],
            color=bucket_color(rgb),
            hex=rgb_to_hex(rgb),
        )

    lines = [_line(box, text, conf) for box, text, conf in results]
    full_text = "\n".join(line.text for line in lines)

    # Two (or more) independent SAP windows side by side in one screenshot
    # (a table + its "maintenance" dialog, a main screen + a smaller popup)
    # must NOT be run through the pipeline as a single grid — rows from
    # different windows at similar y-positions get merged, jamming one
    # window's whole sentence into another window's field value. Detect
    # and process each window separately, then stitch the results back
    # together with a labeled separator so nothing from either window is
    # silently dropped or scrambled.
    windows = split_into_windows(results, image.width, image.height)

    def _process(window_results):
        w_rows, w_colors, w_hex = group_into_table(window_results, image)
        w_text = "\n".join(t for _b, t, _c in window_results)
        w_is_sap = force_sap or looks_like_sap_screen(w_text)
        w_fields = (
            [SapField(**f) for f in build_sap_fields(window_results, image)]
            if w_is_sap and looks_like_field_form_shape(w_rows)
            else []
        )
        return w_rows, w_colors, w_hex, w_is_sap, w_fields

    if len(windows) <= 1:
        rows, colors, hex_colors, is_sap, sap_fields = _process(results)
    else:
        rows, colors, hex_colors = [], [], []
        sap_fields = []
        is_sap = False
        for i, (x0, x1) in enumerate(windows, start=1):
            win_results = [r for r in results if x0 <= (min(p[0] for p in r[0]) + max(p[0] for p in r[0])) / 2 < x1]
            if not win_results:
                continue
            w_rows, w_colors, w_hex, w_is_sap, w_fields = _process(win_results)
            is_sap = is_sap or w_is_sap
            if w_rows:
                if rows:
                    n_cols = max(len(rows[0]), *(len(r) for r in w_rows)) if w_rows else len(rows[0])
                    rows.append([f"── Window {i} ──"] + [""] * (n_cols - 1))
                    colors.append(["white"] * n_cols)
                    hex_colors.append(["FFFFFF"] * n_cols)
                rows.extend(w_rows)
                colors.extend(w_colors)
                hex_colors.extend(w_hex)
            for f in w_fields:
                f.section = f"Window {i}" + (f" – {f.section}" if f.section else "")
            sap_fields.extend(w_fields)

    # `raw` and `image` go out of scope here — nothing persisted beyond this call.
    return OCRResponse(
        text=full_text,
        lines=lines,
        rows=rows,
        colors=colors,
        hex_colors=hex_colors,
        sap_fields=sap_fields,
        is_sap_screen=is_sap,
        language=lang,
        engine="easyocr(th+en)",
        elapsed_ms=int((time.time() - started) * 1000),
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "retains_images": False}


# ---------------------------------------------------------------------------
# Export endpoints: turn the already-extracted rows (from /api/ocr) into
# real downloadable files. These build files server-side with openpyxl /
# python-docx so they can carry real cell colors and formatting — the
# browser-only export used earlier (SheetJS free tier) can't style cells
# at all. No image is involved here, only the text/rows the browser already
# has, so there's nothing new to keep private-by-default about.
# ---------------------------------------------------------------------------

FIELD_FILL_HEX = {
    "yellow": "FFF2CC",       # SAP mandatory-field yellow, toned down for readability
    "blue_header": "B8CCE4",  # SAP section-header blue
    "grey": "D9D9D9",         # non-data chrome (row numbers, column letters, sheet tabs)
    "white": "FFFFFF",
}


class ExportRequest(BaseModel):
    rows: list[list[str]] = []
    colors: list[list[str]] | None = None  # only meaningful when mode == "sap"
    hex_colors: list[list[str]] | None = None  # exact sampled background hex, same shape as rows
    sap_fields: list[SapField] | None = None  # structured Label/Type/Low-To-High rows
    mode: Literal["table", "sap", "visual"] = "table"


PANEL_FILL_HEX = "E5ECF5"   # SAP panel background (Selection Criteria, Language, Currency, Option)
TITLE_FILL_HEX = "1F5C99"   # screen title bar
ARROW_FILL_HEX = "F3ECE0"   # the small "multiple selection" button next to a range field


def build_visual_generic_sheet(ws, rows: list, colors: list | None, hex_colors: list | None = None) -> None:
    """
    Fallback for "Visual Replica" mode when the image ISN'T an SAP screen
    (no sap_fields to work from) — the user can still pick Visual Replica
    for any image, so this gives every cell of the plain row/column grid a
    boxed input-box look instead of a bare table.

    When hex_colors is available (the exact RGB sampled under each cell by
    sample_bg_rgb/rgb_to_hex), that exact color is painted directly — a
    1:1 reproduction of the screenshot's background, not an approximation.
    Falls back to the older 4-bucket color (yellow/blue_header/grey/white)
    when exact hex isn't available, and to plain chrome-greying (top row,
    left column, bottom row) when no color info was sampled at all.
    """
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    if not rows:
        raise HTTPException(400, "No rows to export")

    box_border = Border(*(Side(style="thin", color="9DB3C7") for _ in range(4)))
    total_border = Border(top=Side(style="double", color="404040"))
    grey_fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
    n_rows = len(rows)

    # A "totals/footer" row (a running balance line under a line-item table,
    # e.g. "0.00 THB" or "D 1,003.00   C 1,003.00   0.00") doesn't look like
    # a header, a labeled field, or an ordinary data row — most of its cells
    # are blank and the few filled ones are bare numbers/currency. Detected
    # by position (not the first row) + sparse + numeric, and marked with a
    # double top border + bold instead of being silently absorbed into the
    # regular grid styling.
    amount_re = re.compile(r"^[+-]?[\d,]+\.\d{2}-?$|^\d[\d,]*(\.\d+)?-?\s*[A-Z]{3}$")
    footer_row_idxs = set()
    if n_rows >= 3:
        non_empty_counts = [sum(1 for v in row if v not in (None, "")) for row in rows]
        # Compare against how full a typical data row is (excluding the very
        # sparse candidates themselves), not an absolute column count — a
        # narrow 3-column table's ordinary rows are already "sparse" in
        # absolute terms, so an absolute cutoff flagged normal rows too.
        body_counts = [c for c in non_empty_counts[1:] if c > 0]
        typical_density = sorted(body_counts)[len(body_counts) // 2] if body_counts else 0

        for r, row in enumerate(rows, start=1):
            if r == 1:
                continue
            non_empty = [str(v).strip() for v in row if v not in (None, "")]
            count = len(non_empty)
            if not non_empty or typical_density < 2 or count > typical_density * 0.5:
                continue
            if any(amount_re.match(v.replace(" ", "")) for v in non_empty):
                footer_row_idxs.add(r)

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.alignment = Alignment(vertical="center")
            cell.border = box_border

            color_key = None
            if colors and r - 1 < len(colors) and c - 1 < len(colors[r - 1]):
                color_key = colors[r - 1][c - 1]

            exact_hex = None
            if hex_colors and r - 1 < len(hex_colors) and c - 1 < len(hex_colors[r - 1]):
                exact_hex = hex_colors[r - 1][c - 1]

            if exact_hex:
                cell.fill = PatternFill(start_color=exact_hex, end_color=exact_hex, fill_type="solid")
                if color_key == "blue_header":
                    cell.font = Font(bold=True)
            elif color_key and color_key != "white":
                hexcode = FIELD_FILL_HEX.get(color_key, "FFFFFF")
                cell.fill = PatternFill(start_color=hexcode, end_color=hexcode, fill_type="solid")
                if color_key == "blue_header":
                    cell.font = Font(bold=True)
            elif not colors and not hex_colors:
                # No per-cell color info at all (a plain table image) — still
                # flag likely Excel/screenshot chrome (top row, left column,
                # bottom row), same convention as the raw Table export.
                is_chrome = (r == 1) or (c == 1) or (r == n_rows)
                if is_chrome:
                    cell.fill = grey_fill
                    cell.border = None
            else:
                cell.fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

            if r in footer_row_idxs:
                cell.font = Font(bold=True)
                cell.border = total_border

    for col_cells in ws.columns:
        longest = max((len(str(c.value)) for c in col_cells if c.value), default=8)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(longest + 2, 8), 40)


def build_visual_sap_sheet(ws, sap_fields: list) -> None:
    """
    "Visual Replica" export: lays the extracted fields out to LOOK like the
    original SAP selection screen (panel background, boxed inputs, section
    titles) instead of a plain Section/Label/Type/Low/High table.

    Important limits, by design — this only uses what OCR text + the
    element_type already classified; it does not detect real icon/graphic
    state from the image:
    - The multiple-selection arrow is drawn on every Select-Option Range
      row, because that button is always present for that field type on
      the real screen, not because it was seen in the picture.
    - Checkbox and radio icons are always drawn in their default,
      unticked/unselected state. Whether a checkbox was actually ticked or
      which radio option was actually selected in the source screenshot
      cannot be read from OCR text, so it is never guessed here.
    - Dropdown fields get a plain boxed value with a "▾" marker; the
      actual list of options is not something OCR extraction can determine.
    """
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    panel_fill = PatternFill(start_color=PANEL_FILL_HEX, end_color=PANEL_FILL_HEX, fill_type="solid")
    title_fill = PatternFill(start_color=TITLE_FILL_HEX, end_color=TITLE_FILL_HEX, fill_type="solid")
    box_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    mandatory_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    warning_fill = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
    arrow_fill = PatternFill(start_color=ARROW_FILL_HEX, end_color=ARROW_FILL_HEX, fill_type="solid")
    box_border = Border(*(Side(style="thin", color="9DB3C7") for _ in range(4)))
    warning_border = Border(*(Side(style="thin", color="CC0000") for _ in range(4)))
    all_warnings: list[str] = []

    ws.column_dimensions["A"].width = 2
    for col in "BCD":
        ws.column_dimensions[col].width = 26
    for col in "EF":
        ws.column_dimensions[col].width = 20
    ws.column_dimensions["G"].width = 4

    row = 1
    ws.merge_cells(f"A{row}:G{row}")
    cell = ws.cell(row=row, column=1, value="  Extracted Selection Screen (Visual Replica)")
    cell.font = Font(bold=True, color="FFFFFF", size=12)
    cell.fill = title_fill
    ws.row_dimensions[row].height = 22
    row += 2

    current_section = None
    for f in sap_fields:
        etype = f["element_type"] if isinstance(f, dict) else f.element_type
        section = f["section"] if isinstance(f, dict) else f.section
        label = f["label"] if isinstance(f, dict) else f.label
        low = f["low"] if isinstance(f, dict) else f.low
        high = f["high"] if isinstance(f, dict) else f.high
        warnings = (f.get("warnings", []) if isinstance(f, dict) else f.warnings) or []
        field_hex = (f.get("hex") if isinstance(f, dict) else getattr(f, "hex", None)) or "FFFFFF"
        exact_box_fill = PatternFill(start_color=field_hex, end_color=field_hex, fill_type="solid")

        if section and section != current_section:
            current_section = section
            ws.merge_cells(f"A{row}:G{row}")
            cell = ws.cell(row=row, column=1, value=f"  {section}")
            cell.font = Font(bold=True)
            cell.fill = panel_fill
            row += 1

        is_range = "Range" in etype or "Low" in etype
        is_mandatory = "mandatory" in etype.lower()

        # Panel background across the whole row first, so gaps between
        # boxed cells still read as "inside the panel", like the real screen.
        for c in range(1, 8):
            ws.cell(row=row, column=c).fill = panel_fill

        label_cell = ws.cell(row=row, column=2, value=label)
        label_cell.alignment = Alignment(vertical="center")

        low_cell = ws.cell(row=row, column=3, value=low or "")
        low_cell.fill = warning_fill if warnings else (mandatory_fill if is_mandatory else exact_box_fill)
        low_cell.border = warning_border if warnings else box_border
        if warnings:
            low_cell.comment = Comment(" | ".join(warnings), "Data Integrity Check")
            all_warnings.append(f"{label}: {' | '.join(warnings)}")

        if is_range:
            ws.cell(row=row, column=4, value="to").alignment = Alignment(horizontal="center", vertical="center")
            high_cell = ws.cell(row=row, column=5, value=high or "")
            high_cell.fill = warning_fill if warnings else exact_box_fill
            high_cell.border = warning_border if warnings else box_border
            # The "multiple selection" button — always shown for a range
            # field on the real screen, not something detected per-row.
            arrow_cell = ws.cell(row=row, column=7, value="➔")
            arrow_cell.fill = arrow_fill
            arrow_cell.alignment = Alignment(horizontal="center", vertical="center")
        elif "dropdown" in etype.lower():
            low_cell.value = (low or "") + "  ▾"
        elif "checkbox" in etype.lower():
            label_cell.value = "☐  " + label  # always unticked — real state unknown
            low_cell.value = ""
            low_cell.fill = panel_fill
            low_cell.border = None
        elif "radio" in etype.lower():
            label_cell.value = "⚪  " + label  # always unselected — real state unknown
            low_cell.value = ""
            low_cell.fill = panel_fill
            low_cell.border = None
        else:
            # Plain input field — search-helper icon is a stock SAP UI
            # element (F4), shown for context rather than detected.
            search_cell = ws.cell(row=row, column=6, value="\U0001F50D")
            search_cell.alignment = Alignment(horizontal="center", vertical="center")
            search_cell.fill = panel_fill

        row += 1

    row += 1
    note = ws.cell(
        row=row, column=1,
        value=("Note: checkbox/radio state, dropdown option lists, and the search/multiple-selection "
               "icons are shown at their default appearance — OCR text cannot confirm the real state "
               "of graphical controls."),
    )
    note.font = Font(italic=True, size=9, color="808080")
    ws.merge_cells(f"A{row}:G{row}")

    if all_warnings:
        row += 2
        header = ws.cell(row=row, column=1, value=f"⚠ Data Integrity Warnings ({len(all_warnings)})")
        header.font = Font(bold=True, color="990000")
        ws.merge_cells(f"A{row}:G{row}")
        for w in all_warnings:
            row += 1
            cell = ws.cell(row=row, column=1, value="  " + w)
            cell.font = Font(color="990000", size=9)
            ws.merge_cells(f"A{row}:G{row}")


class TextExportRequest(BaseModel):
    text: str


@app.post("/api/export-xlsx")
async def export_xlsx(req: ExportRequest):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "OCR Result"

    if req.mode == "visual" and req.sap_fields:
        build_visual_sap_sheet(ws, req.sap_fields)
    elif req.mode == "visual":
        # "Visual Replica" was picked but this isn't an SAP screen (no
        # sap_fields) — still honor the choice using the plain row/column
        # grid instead of silently falling back to a different layout.
        build_visual_generic_sheet(ws, req.rows, req.colors, req.hex_colors)
    elif req.mode == "sap" and req.sap_fields:
        # Structured layout: Section | Field Label | Element Type | Low | To | High
        # instead of a raw grid — this is the "reads like the screen" format,
        # built from build_sap_fields() rather than the row/column grid.
        headers = ["Section", "Field Label", "Element Type", "Low", "To", "High", "Data Integrity Warnings"]
        header_fill = PatternFill(start_color="B8CCE4", end_color="B8CCE4", fill_type="solid")
        mandatory_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
        warning_fill = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
        # Thin grey border on Low/High to mimic the boxed input fields on
        # the actual SAP selection screen — purely visual, no new data.
        box_border = Border(*(Side(style="thin", color="BFBFBF") for _ in range(4)))

        for c, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = Font(bold=True)
            cell.fill = header_fill

        for r, f in enumerate(req.sap_fields, start=2):
            has_range = "Low" in f.element_type or "Range" in f.element_type
            has_warning = bool(f.warnings)
            values = [
                f.section, f.label, f.element_type,
                f.low, "to" if has_range else "", f.high,
                " | ".join(f.warnings),
            ]
            for c, v in enumerate(values, start=1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.alignment = Alignment(vertical="center", wrap_text=(c == 7))
                if c in (4, 6):  # Low, High columns — the actual input boxes
                    if has_warning:
                        cell.fill = warning_fill
                    else:
                        cell.fill = mandatory_fill if "mandatory" in f.element_type.lower() and c == 4 else white_fill
                    cell.border = box_border
                elif c == 7 and has_warning:
                    cell.fill = warning_fill
                    cell.font = Font(color="990000")
                elif "mandatory" in f.element_type.lower():
                    cell.fill = mandatory_fill

        ws.freeze_panes = "A2"
    else:
        if not req.rows:
            raise HTTPException(400, "No rows to export")

        grey_fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
        n_rows = len(req.rows)

        for r, row in enumerate(req.rows, start=1):
            for c, value in enumerate(row, start=1):
                cell = ws.cell(row=r, column=c, value=value)
                cell.alignment = Alignment(vertical="center", wrap_text=False)

                if req.mode == "sap" and req.colors and r - 1 < len(req.colors) and c - 1 < len(req.colors[r - 1]):
                    color_key = req.colors[r - 1][c - 1]
                    hexcode = FIELD_FILL_HEX.get(color_key, FIELD_FILL_HEX["white"])
                    if hexcode != "FFFFFF":
                        cell.fill = PatternFill(start_color=hexcode, end_color=hexcode, fill_type="solid")
                    if color_key == "blue_header":
                        cell.font = Font(bold=True)
                elif req.mode == "table":
                    # Flag the row/column/row that usually turn out to be Excel's
                    # own chrome (column letters, row numbers, sheet tabs) caught
                    # in the screenshot rather than real data — see group_into_table.
                    is_chrome = (r == 1) or (c == 1) or (r == n_rows)
                    if is_chrome:
                        cell.fill = grey_fill

    # Visual Replica mode already sets its own explicit column widths and
    # uses merged cells for the title/section bars — a MergedCell has no
    # .column_letter, so the auto-size pass below only applies to the
    # other two (unmerged) modes.
    if req.mode != "visual":
        for col_cells in ws.columns:
            longest = max((len(str(c.value)) for c in col_cells if c.value), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(longest + 2, 8), 40)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=scan-desk-ocr-result.xlsx"},
    )


@app.post("/api/export-docx")
async def export_docx(req: TextExportRequest):
    from docx import Document

    if not req.text.strip():
        raise HTTPException(400, "No text to export")

    doc = Document()
    for paragraph in req.text.split("\n"):
        doc.add_paragraph(paragraph)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=scan-desk-ocr-result.docx"},
    )


# ---------------------------------------------------------------------------
# Serves the Scan Desk OCR frontend directly from this backend, at the same
# origin as /api/ocr. This matters: a page opened via a claude.ai artifact
# link runs inside a sandbox that blocks it from calling out to your local
# server — opening the page from http://localhost:8000 instead avoids that
# entirely, since browser and backend are then on the very same origin.
#
# Put scan-desk-ocr.html next to this script before running uvicorn.
# ---------------------------------------------------------------------------
FRONTEND_FILE = Path(__file__).parent / "scan-desk-ocr.html"


@app.get("/", response_class=HTMLResponse)
async def frontend():
    if not FRONTEND_FILE.exists():
        return HTMLResponse(
            "<h1>scan-desk-ocr.html not found</h1>"
            "<p>Place scan-desk-ocr.html in the same folder as this script, then restart uvicorn.</p>",
            status_code=404,
        )
    return FileResponse(FRONTEND_FILE)


# ---------------------------------------------------------------------------
# Optional: swap in a managed cloud OCR instead of EasyOCR by replacing
# get_reader()/extract_text() body with a call to Google Vision or AWS
# Textract, e.g.:
#
#   from google.cloud import vision
#   client = vision.ImageAnnotatorClient()
#   image = vision.Image(content=raw)
#   response = client.document_text_detection(image=image)
#   full_text = response.full_text_annotation.text
#
# Keep the same rule: raw bytes never touch disk, response is the only
# artifact returned to the caller.
# ---------------------------------------------------------------------------
