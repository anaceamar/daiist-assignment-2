"""
Assignment 3: build the index (`main.py index`).

Corpus: EU passenger rights law (air, rail, bus/coach, sea) from the EU
Publications Office. Pipeline per file:

    extract_lines(path)   HTML/XHTML/PDF/TXT  -> clean text lines, in order
    split_units(lines)    lines               -> legal units (Recital 5, Article 7,
                                                 Annex I, Section 4.3.2 ...)
    chunk_units(units)    units               -> chunks for one design
    embed_and_store(...)  chunks              -> vectors saved in index/vectors.jsonl

Two chunk designs are built so evaluate.py can compare them:
    fixed      ~250-word sliding windows (50-word overlap), blind to structure
    structure  one chunk per legal unit; long units split at paragraph
               boundaries (<= 300 words), each chunk headed by its document and
               unit title

Every vector is appended to index/vectors.jsonl the moment it arrives, keyed by
a hash of (model, text), so a crash or the free-tier daily limit loses nothing
and a re-run embeds only chunks without a saved vector.

server.py's add_document tool imports `add_file()` from here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "index"))
VECTORS_FILE = INDEX_DIR / "vectors.jsonl"
DOCS_FILE = INDEX_DIR / "documents.json"
CONFIG_FILE = INDEX_DIR / "config.json"

from settings import EMBED_MODEL  # noqa: E402 - gemini-embedding-2 unless overridden in .env
EMBED_DIM = 768            # truncated Matryoshka size: 4x smaller index, little quality loss
EMBED_BATCH = 50           # texts per request (each text counts towards the daily quota)
DESIGNS = ("fixed", "structure")

FIXED_WORDS, FIXED_OVERLAP = 250, 50
STRUCT_MAX_WORDS = 300

CELLAR = "http://publications.europa.eu/resource/celex/"

# The corpus. `source` is the id used in chunk metadata and eval/questions.csv.
DOCS = [
    {"source": "air-261-2004", "celex": "32004R0261", "mode": "air", "kind": "regulation",
     "title": "Regulation (EC) No 261/2004 - air passenger rights (denied boarding, cancellation, delay)"},
    {"source": "air-guidelines-2024", "celex": "52024XC05687", "mode": "air", "kind": "guidelines",
     "title": "Commission Interpretative Guidelines on Regulation 261/2004 (C/2024/5687)"},
    {"source": "rail-2021-782", "celex": "32021R0782", "mode": "rail", "kind": "regulation",
     "title": "Regulation (EU) 2021/782 - rail passengers' rights and obligations"},
    {"source": "bus-181-2011", "celex": "32011R0181", "mode": "bus", "kind": "regulation",
     "title": "Regulation (EU) No 181/2011 - bus and coach passenger rights"},
    {"source": "sea-1177-2010", "celex": "32010R1177", "mode": "sea", "kind": "regulation",
     "title": "Regulation (EU) No 1177/2010 - rights of passengers travelling by sea and inland waterway"},
]


def log(*args) -> None:
    """Log to stderr: server.py imports this module and stdout carries MCP."""
    print(*args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- download

def download(celex: str, out: Path) -> Path:
    """Fetch a document from the Publications Office (CELLAR) by CELEX number.

    EUR-Lex's own site answers scripts with an empty bot challenge, so we ask
    CELLAR directly: newer acts exist as XHTML, older ones (261/2004) as HTML.
    """
    if out.exists() and out.stat().st_size > 1000:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = CELLAR + urllib.parse.quote(celex)
    last_error = None
    for accept in ("application/xhtml+xml", "text/html"):
        request = urllib.request.Request(url, headers={"Accept": accept, "Accept-Language": "eng"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                out.write_bytes(response.read())
                log(f"  downloaded {celex} ({accept}) -> {out.name}")
                return out
        except Exception as error:  # noqa: BLE001 - try the next format
            last_error = error
    raise RuntimeError(f"Could not download {celex}: {last_error}")


# --------------------------------------------------------------------------- extract

def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("­", "")
    text = re.sub(r"\(\s*\)", "", text)                 # empty footnote markers left by removed tags
    text = re.sub(r"\[\d+\]", "", text)                 # old-style footnote markers: [1]
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_lines(path: Path) -> list[str]:
    """Text lines in reading order. Tables are flattened one row per line, so a
    recital number and its text (separate cells in the Official Journal layout)
    end up on one line: '(14) Rights of bus ...'."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        lines = []
        for number, page in enumerate(PdfReader(str(path)).pages, 1):
            lines.append(f"[[PAGE {number}]]")
            lines += [clean(l) for l in (page.extract_text() or "").splitlines() if clean(l)]
        return lines
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [clean(p) for p in re.split(r"\n\s*\n", text) if clean(p)]

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    # Footnote markers and footnote bodies: case-law references that would
    # otherwise land in the middle of sentences and in their own fake chunks.
    for element in soup.select(".oj-note-tag, .oj-note, script, style, .note"):
        element.decompose()
    body = soup.body or soup
    lines = []
    for element in body.find_all(["p", "tr", "h1", "h2", "h3", "h4", "li"]):
        if element.name != "tr" and element.find_parent(["table", "li"]) is not None and element.name != "li":
            continue
        if element.name == "li" and element.find_parent("li") is not None:
            continue
        if element.name == "tr" and (element.find("table") is not None):
            continue  # a row that wraps a nested table: its inner rows are read instead
        text = clean(element.get_text(" "))
        if text:
            lines.append(text)
    return lines


# --------------------------------------------------------------------------- units

@dataclass
class Unit:
    location: str          # 'Article 7', 'Recital 12', 'Annex I', 'Annex I, Article 32', 'Section 4.3.2'
    title: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(l.split()) for l in self.lines)


ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)$")
ANNEX = re.compile(r"^ANNEX\s*([IVX]+)?$", re.I)
RECITAL = re.compile(r"^\((\d+)\)\s+(.+)$")
SECTION = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(\S.*?)$")
TOC_ENTRY = re.compile(r"^(\d+(?:\.\d+)*\.|[A-Z]\.)\s+.+\s\d{1,3}$")
SIGNATURE = re.compile(r"^Done at [A-Z][a-z]+")
DIVISION = re.compile(r"^(CHAPTER|TITLE|SECTION|PART)\s+[IVXLC\d]+$")


def split_units(lines: list[str], kind: str | None = None) -> list[Unit]:
    if kind is None:
        if any(ARTICLE.match(l) for l in lines):
            kind = "regulation"
        elif any(l.startswith("[[PAGE") for l in lines):
            kind = "pdf"
        else:
            kind = "plain"
    if kind == "regulation":
        return _regulation_units(lines)
    if kind == "guidelines":
        return _guideline_units(lines)
    if kind == "pdf":
        return _page_units(lines)
    return _plain_units(lines)


def _regulation_units(lines: list[str]) -> list[Unit]:
    """Recitals (after 'Whereas:'), articles, annexes. The signature block
    ('Done at ...') and the footnotes after it are dropped; annexes resume.
    Articles inside an annex (2021/782 Annex I reproduces the CIV rules, whose
    articles reuse the numbers 1-60) are labelled 'Annex I, Article N' so they
    are not confused with the regulation's own Article N."""
    units: list[Unit] = []
    state = "preamble"
    annex = None
    current: Unit | None = None
    expect_title = skip_caps = False

    def start(location: str) -> Unit:
        unit = Unit(location)
        units.append(unit)
        return unit

    for line in lines:
        if line.lower().startswith("whereas"):
            state = "recitals"
            continue
        if SIGNATURE.match(line):
            state, current = "signature", None
            continue
        if line.upper().startswith("HAVE ADOPTED THIS"):      # end of the recitals
            state, current = "enacting", None
            continue
        annex_match = ANNEX.match(line)
        if annex_match:
            annex = f"Annex {annex_match.group(1) or ''}".strip()
            state, current, expect_title = "body", start(annex), True
            continue
        article_match = ARTICLE.match(line)
        if article_match and state != "signature":
            label = f"Article {article_match.group(1)}"
            state, current, expect_title = "body", start(f"{annex}, {label}" if annex else label), True
            continue
        if state == "recitals":
            recital = RECITAL.match(line)
            if recital:
                current = start(f"Recital {recital.group(1)}")
                current.lines.append(recital.group(2))
            elif current is not None:
                current.lines.append(line)
            continue
        if state != "body" or current is None:
            continue
        # Chapter headings ('CHAPTER V' + an all-caps title) belong to no article;
        # left in, they were glued to the end of the article before them.
        if DIVISION.match(line):
            skip_caps = True
            continue
        if skip_caps and line.isupper():
            skip_caps = False
            continue
        skip_caps = False
        if expect_title and len(line.split()) <= 14 and not re.match(r"^(\d+\.|\([a-z0-9]+\))", line):
            current.title = line
            expect_title = False
            continue
        expect_title = False
        current.lines.append(line)
    return [u for u in units if u.lines]


def _guideline_units(lines: list[str]) -> list[Unit]:
    """Numbered sections ('4.3.2. Title'). The table of contents repeats every
    heading followed by a page number; we skip it until the first heading
    appears again without one."""
    units: list[Unit] = []
    current: Unit | None = None
    in_toc, first_heading = False, None
    for line in lines:
        if line.upper().startswith("TABLE OF CONTENTS"):
            in_toc = True
            continue
        if in_toc:
            if first_heading is None and TOC_ENTRY.match(line):
                first_heading = re.sub(r"\s\d{1,3}$", "", line)
                continue
            if first_heading is not None and line == first_heading:
                in_toc = False
            else:
                continue
        section = SECTION.match(line)
        if section and len(line.split()) <= 40 and not line.endswith("."):
            current = Unit(f"Section {section.group(1)}", section.group(2))
            units.append(current)
            continue
        if current is not None:
            current.lines.append(line)
    return [u for u in units if u.lines]


def _page_units(lines: list[str]) -> list[Unit]:
    units: list[Unit] = []
    for line in lines:
        page = re.match(r"^\[\[PAGE (\d+)\]\]$", line)
        if page:
            units.append(Unit(f"Page {page.group(1)}"))
        elif units:
            units[-1].lines.append(line)
    return [u for u in units if u.lines]


def _plain_units(lines: list[str]) -> list[Unit]:
    return [Unit(f"Paragraph {i}", lines=[l]) for i, l in enumerate(lines, 1)]


# --------------------------------------------------------------------------- chunks

@dataclass
class Chunk:
    id: str
    source: str
    location: str          # where the chunk starts, for citations
    units: list[str]       # every unit the chunk touches, for hit-rate matching
    mode: str
    title: str             # document title
    text: str              # what is shown to the model
    embed_text: str        # what is embedded (text + document/unit header)
    added: bool = False    # True for documents added later through add_document
    hash: str = ""


def text_hash(text: str) -> str:
    return hashlib.sha256(f"{EMBED_MODEL}|{EMBED_DIM}|{text}".encode()).hexdigest()[:24]


def _header(doc: dict, unit: Unit) -> str:
    return f"{doc['title']} | {unit.location}{' - ' + unit.title if unit.title else ''}"


def chunk_units(units: list[Unit], doc: dict, design: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(location: str, touched: list[str], text: str, header: str) -> None:
        chunk = Chunk(
            id=f"{doc['source']}#{design}#{len(chunks):04d}", source=doc["source"], location=location,
            units=touched, mode=doc["mode"], title=doc["title"], text=text,
            embed_text=f"{header}\n{text}", added=doc.get("added", False),
        )
        chunk.hash = text_hash(chunk.embed_text)
        chunks.append(chunk)

    if design == "structure":
        for unit in units:
            heading = f"{unit.location}{' - ' + unit.title if unit.title else ''}"
            parts, buffer = [], []
            for line in unit.lines:
                if buffer and sum(len(l.split()) for l in buffer) + len(line.split()) > STRUCT_MAX_WORDS:
                    parts.append(buffer)
                    buffer = []
                buffer.append(line)
            if buffer:
                parts.append(buffer)
            for i, part in enumerate(parts):
                suffix = f" (part {i + 1}/{len(parts)})" if len(parts) > 1 else ""
                body = "\n".join(part)
                add(unit.location, [unit.location], f"{heading}{suffix}\n{body}", _header(doc, unit))
        return chunks

    if design == "fixed":
        words: list[tuple[str, str]] = []           # (word, unit location)
        for unit in units:
            words += [(w, unit.location) for w in " ".join(unit.lines).split()]
        step = FIXED_WORDS - FIXED_OVERLAP
        for start in range(0, max(len(words) - FIXED_OVERLAP, 1), step):
            window = words[start:start + FIXED_WORDS]
            touched = list(dict.fromkeys(loc for _, loc in window))
            add(touched[0], touched, " ".join(w for w, _ in window), doc["title"])
        return chunks

    raise ValueError(f"Unknown design {design!r}")


def chunk_file(path: Path, doc: dict, design: str) -> list[Chunk]:
    """The whole per-file pipeline: extract, split into units, chunk."""
    units = split_units(extract_lines(path), doc.get("kind"))
    if not units:
        raise ValueError(f"No text could be extracted from {path.name}.")
    return chunk_units(units, doc, design)


# --------------------------------------------------------------------------- storage

def chunks_file(design: str) -> Path:
    return INDEX_DIR / f"chunks_{design}.jsonl"


def load_chunks(design: str) -> list[dict]:
    path = chunks_file(design)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_chunks(design: str, chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = chunks_file(design).with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    tmp.replace(chunks_file(design))


def load_vectors() -> dict[str, list[float]]:
    vectors = {}
    if VECTORS_FILE.exists():
        for line in VECTORS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash: it is simply embedded again
                vectors[row["h"]] = row["v"]
    return vectors


def load_documents() -> dict:
    return json.loads(DOCS_FILE.read_text(encoding="utf-8")) if DOCS_FILE.exists() else {}


def save_documents(documents: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FILE.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- embeddings

class QuotaExhausted(RuntimeError):
    pass


_client = None


def _genai():
    global _client
    if _client is None:
        from google import genai
        from api_key import load_api_key
        _client = genai.Client(api_key=load_api_key())
    return _client


def _normalise(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5 or 1.0
    return [round(x / norm, 6) for x in vector]


_working_config: dict | None = None     # the request options this model accepted last time


def _configs(task: str) -> list[dict]:
    """Request options to try, most specific first. Models differ in whether they
    accept task types and output_dimensionality; vectors are truncated to
    EMBED_DIM and normalised here either way."""
    return [{"task_type": task, "output_dimensionality": EMBED_DIM}, {"output_dimensionality": EMBED_DIM},
            {"task_type": task}, {}]


def embed_texts(texts: list[str], task: str) -> list[list[float]]:
    """Embed a batch with Gemini, retrying on per-minute limits.

    `task` is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY. Raises QuotaExhausted when
    the daily free-tier quota is spent, so callers can stop and keep progress.
    """
    global _working_config
    if os.environ.get("FAKE_EMBEDDINGS"):          # offline testing only, never used for the real index
        return [_fake_vector(t) for t in texts]
    from google.genai import types
    options = _configs(task)
    if _working_config is not None:
        options = [o for o in options if set(o) == set(_working_config)] + options
    option_index, attempt = 0, 0
    while attempt < 10:
        option = options[option_index]
        try:
            result = _genai().models.embed_content(
                model=EMBED_MODEL, contents=texts, config=types.EmbedContentConfig(**option))
            if len(result.embeddings) != len(texts):
                # Some multimodal models merge a list into one embedding: embed one by one instead.
                return [embed_texts([t], task)[0] for t in texts] if len(texts) > 1 else \
                    [_normalise(list(result.embeddings[0].values)[:EMBED_DIM])]
            _working_config = option
            return [_normalise(list(e.values)[:EMBED_DIM]) for e in result.embeddings]
        except Exception as error:  # noqa: BLE001 - inspect the API's message
            message = str(error)
            if "INVALID_ARGUMENT" in message or "400" in message:
                if option_index + 1 < len(options) and len(texts) <= 100:
                    option_index += 1          # try the request without the option the model rejected
                    continue
                raise
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                if re.search(r"per ?day|PerDay|daily", message, re.I):
                    raise QuotaExhausted(message) from error
                delay = re.search(r"retry in ([\d.]+)s|retryDelay['\": ]+(\d+)s", message)
                wait = float(delay.group(1) or delay.group(2)) + 2 if delay else 20 * (attempt + 1)
                log(f"  rate limited, waiting {wait:.0f}s ...")
                time.sleep(wait)
            elif any(code in message for code in ("500", "503", "UNAVAILABLE", "DEADLINE")):
                time.sleep(10 * (attempt + 1))
            else:
                raise
            attempt += 1
    raise RuntimeError("Gemini embedding kept failing after 10 attempts.")


def _fake_vector(text: str) -> list[float]:
    """Deterministic bag-of-words hash vector, for testing the plumbing offline."""
    vector = [0.0] * EMBED_DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM] += 1.0
    return _normalise(vector)


def embed_and_store(chunks: list[dict]) -> int:
    """Embed every chunk without a saved vector; append each batch to
    vectors.jsonl as soon as it arrives. Returns how many were embedded."""
    saved = load_vectors()
    todo, seen = [], set()
    for chunk in chunks:
        if chunk["hash"] not in saved and chunk["hash"] not in seen:
            todo.append(chunk)
            seen.add(chunk["hash"])
    if not todo:
        return 0
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    with VECTORS_FILE.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            vectors = embed_texts([c["embed_text"] for c in batch], "RETRIEVAL_DOCUMENT")
            for chunk, vector in zip(batch, vectors):
                out.write(json.dumps({"h": chunk["hash"], "v": vector}) + "\n")
            out.flush()
            done += len(batch)
            log(f"  embedded {done}/{len(todo)}")
    return done


# --------------------------------------------------------------------------- add a document

def doc_for_file(path: Path) -> dict:
    """Metadata for a file that is not in DOCS: id from the file name, mode guessed from its text."""
    for doc in DOCS:
        if doc["source"] == path.stem:
            return dict(doc)
    text = " ".join(extract_lines(path)[:200]).lower()
    counts = {mode: sum(text.count(w) for w in words) for mode, words in {
        "air": ("air carrier", "flight", "airport"), "rail": ("railway", "train", "rail"),
        "bus": ("bus", "coach"), "sea": ("ship", "port", "waterway", "vessel")}.items()}
    head = extract_lines(path)[:60]
    title = next((l for l in head if re.match(r"^(REGULATION|DIRECTIVE|DECISION|COMMISSION|Regulation|Directive)\b", l)),
                 next((l for l in head if len(l.split()) > 6), path.stem))[:160]
    return {"source": path.stem, "mode": max(counts, key=counts.get), "kind": None, "title": title}


def search_bases() -> tuple[str, ...]:
    """The chunk design(s) the server searches. Added documents are indexed only
    there: embedding them in a design nobody searches would waste quota."""
    try:
        design = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("search_design", "structure-hybrid")
    except FileNotFoundError:
        design = "structure-hybrid"
    return ("fixed",) if design.startswith("fixed") else ("structure",)


def add_file(path: str | Path, added: bool = True) -> dict:
    """Run one file through the pipeline and store it in the design the server searches.

    If the same file (same SHA-256) is already indexed, nothing happens. If a
    different version with the same name is indexed, its chunks are replaced.
    Returns a summary dict.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() not in (".html", ".htm", ".xhtml", ".xml", ".pdf", ".txt", ".md"):
        raise ValueError(f"Unsupported file type {path.suffix}; use HTML, PDF, TXT or MD.")
    documents = load_documents()
    doc = doc_for_file(path)
    doc["added"] = added
    sha = file_sha(path)
    previous = documents.get(doc["source"])
    if previous and previous.get("sha256") == sha:
        return {"status": "already_indexed", "source": doc["source"], "chunks": previous.get("chunks", {})}

    counts = {}
    for design in search_bases():
        new = [asdict(c) for c in chunk_file(path, doc, design)]
        embed_and_store(new)        # embed first: if the quota runs out, the old version stays searchable
        kept = [c for c in load_chunks(design) if c["source"] != doc["source"]]
        save_chunks(design, kept + new)
        counts[design] = len(new)
    documents[doc["source"]] = {
        "title": doc["title"], "mode": doc["mode"], "file": path.name, "sha256": sha,
        "added": added, "chunks": counts, "words": sum(len(l.split()) for l in extract_lines(path)),
    }
    save_documents(documents)
    return {"status": "replaced" if previous else "added", "source": doc["source"], "chunks": counts}


# --------------------------------------------------------------------------- main

def build() -> int:
    documents = load_documents()
    all_chunks: dict[str, list[dict]] = {}
    for doc in DOCS:
        path = download(doc["celex"], CORPUS_DIR / f"{doc['source']}.html")
        units = split_units(extract_lines(path), doc["kind"])
        words = sum(u.words for u in units)
        for design in DESIGNS:
            all_chunks.setdefault(design, []).extend(asdict(c) for c in chunk_units(units, doc, design))
        documents[doc["source"]] = {
            "title": doc["title"], "mode": doc["mode"], "file": path.name, "sha256": file_sha(path),
            "added": False, "celex": doc["celex"],
            "url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{doc['celex']}",
            "units": len(units), "words": words,
            "chunks": {d: sum(1 for c in all_chunks[d] if c["source"] == doc["source"]) for d in DESIGNS},
        }
        log(f"{doc['source']}: {len(units)} units, {words} words")

    # Keep documents added later through add_document.
    for design in DESIGNS:
        extra = [c for c in load_chunks(design) if c.get("added")]
        save_chunks(design, all_chunks[design] + extra)
    save_documents(documents)
    previous = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
    CONFIG_FILE.write_text(json.dumps({
        "search_design": previous.get("search_design", "structure-hybrid"),
        "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM,
        "designs": {"fixed": f"{FIXED_WORDS}-word windows, {FIXED_OVERLAP} overlap",
                    "structure": f"legal units, split at paragraphs above {STRUCT_MAX_WORDS} words"},
    }, indent=2), encoding="utf-8")

    total_words = sum(d["words"] for s, d in documents.items() if not d.get("added"))
    log(f"\nCorpus: {len(DOCS)} documents, {total_words} words")
    for design in DESIGNS:
        log(f"  {design}: {len(all_chunks[design])} chunks")

    try:
        for design in ("structure", "fixed"):      # the chosen design first, in case the quota runs out
            log(f"\nEmbedding '{design}' chunks without a saved vector ...")
            log(f"  {embed_and_store(all_chunks[design])} new vectors")
    except QuotaExhausted:
        log("\nThe daily embedding quota is spent. Everything embedded so far is saved;\n"
            "run `uv run python main.py index` again after it resets (09:00 in Spain).")
        return 1
    log("\nIndex complete.")
    return 0


if __name__ == "__main__":
    sys.exit(build())
