"""Book RAG: BM25 finds candidates, Jev picks the relevant passages, LiteLLM writes, Jev verifies claims."""

import json
import math
import os
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv()

BOOKS_DIR = Path(__file__).parent / "books"
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")
CANDIDATES = 30  # BM25 shortlist sent to Jev
TOP_K = 6  # passages given to the writer
MIN_RELEVANCE = 0.5  # Jev p(yes) to keep a passage
MIN_SUPPORT = 0.7  # Jev p(yes) to keep a claim
MAX_UPLOAD = 100 * 1024 * 1024

STOP = set("the a an and or of to in is it that for on with as be are this was i you my me how can do what why".split())


def tokens(text):
    # ponytail: crude plural stripping instead of a stemmer; swap in nltk if recall suffers
    return [w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]


def load_passages(books_dir=BOOKS_DIR, words=250):
    passages = []
    for path in sorted(books_dir.glob("*.txt")):
        chunk = []
        for para in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore")):
            w = para.split()
            for j in range(0, len(w), words):  # hard-split long paragraphs (PDF text often has no blank lines)
                chunk.append(" ".join(w[j : j + words]))
                if sum(len(p.split()) for p in chunk) >= words:
                    passages.append({"book": path.stem, "text": "\n\n".join(chunk)})
                    chunk = []
        if chunk:
            passages.append({"book": path.stem, "text": "\n\n".join(chunk)})
    return passages


def save_book(filename, data, books_dir=BOOKS_DIR):
    """Store an uploaded .pdf or .txt as books/<name>.txt. Returns the book title."""
    path = Path(filename)
    title = re.sub(r"[^\w .,'()&-]", "", path.stem).strip()[:120]
    ext = path.suffix.lower()
    if not title or ext not in (".pdf", ".txt"):
        raise ValueError("Upload a .pdf or .txt file.")
    if ext == ".pdf":
        from io import BytesIO
        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    if len(text.split()) < 50:
        raise ValueError(f"No readable text in {filename}. Scanned PDFs (page images) aren't supported.")
    books_dir.mkdir(exist_ok=True)
    (books_dir / f"{title}.txt").write_text(text, encoding="utf-8")
    return title


class BM25:
    def __init__(self, passages, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(p["text"])) for p in passages]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k):
        q = [w for w in set(tokens(query)) if w in self.idf]
        scores = []
        for i, (doc, length) in enumerate(zip(self.docs, self.lens)):
            s = sum(
                self.idf[w] * doc[w] * (self.k1 + 1) / (doc[w] + self.k1 * (1 - self.b + self.b * length / self.avg))
                for w in q
                if w in doc
            )
            if s > 0:
                scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]


# --- Jev (TypeSafe System One) ---

_jev = None


def jev(state, questions):
    """Return {name: p(yes)} for a dict of yes/no questions about `state`."""
    global _jev
    from typesafe_sdk import Noul, TypeSafeClient

    _jev = _jev or TypeSafeClient()
    resp = _jev.system_one(state=state, questions={k: Noul(**v) for k, v in questions.items()})
    return {k: a.noul for k, a in resp.nouls.items()}


def jev_select(question, passages):
    """Job 1: keep the passages Jev thinks actually help answer the question."""
    state = {"question": question, "passages": {f"p{i}": p["text"] for i, p in enumerate(passages)}}
    probs = jev(state, {
        f"p{i}": {
            "instructions": f"Passage p{i} contains ideas or advice that help answer the question.",
            "criteria": {"true": "Directly relevant to the question", "false": "Off-topic or only shares keywords"},
        }
        for i in range(len(passages))
    })
    ranked = sorted(range(len(passages)), key=lambda i: probs.get(f"p{i}", 0), reverse=True)
    return [{**passages[i], "relevance": round(probs[f"p{i}"], 3)} for i in ranked[:TOP_K] if probs[f"p{i}"] >= MIN_RELEVANCE]


def jev_verify(passages, claims):
    """Job 2: score each claim for support by the passages; caller drops the unsupported ones."""
    state = {
        "passages": {f"[{i + 1}]": p["text"] for i, p in enumerate(passages)},
        "claims": {f"c{i}": c for i, c in enumerate(claims)},
    }
    probs = jev(state, {
        f"c{i}": {
            "instructions": f"Claim c{i} is supported by the passages.",
            "criteria": {
                "true": "Everything the claim says is stated or directly implied by the passages",
                "false": "Part of the claim is missing from, or contradicts, the passages",
            },
        }
        for i in range(len(claims))
    })
    return [probs[f"c{i}"] for i in range(len(claims))]


# --- LiteLLM writer ---

def write_claims(question, passages):
    import litellm

    sources = "\n\n".join(f"[{i + 1}] ({p['book']})\n{p['text']}" for i, p in enumerate(passages))
    resp = litellm.completion(model=LLM_MODEL, temperature=0.2, messages=[
        {"role": "system", "content": (
            "Answer using ONLY the numbered book passages. Write 3-8 short, practical claims, "
            "one per line, each starting with '- ' and ending with its citations like [1] or [2][3]. "
            "No intro, no outro, nothing the passages don't say."
        )},
        {"role": "user", "content": f"Passages:\n\n{sources}\n\nQuestion: {question}"},
    ])
    return parse_claims(resp.choices[0].message.content)


def parse_claims(text):
    return [line.lstrip("-*• ").strip() for line in text.splitlines() if line.strip().startswith(("-", "*", "•"))]


# --- pipeline ---

def reload_library():
    global LIBRARY
    passages = load_passages()
    LIBRARY = (passages, BM25(passages))  # one assignment, so requests never see a half-built index


def stats():
    passages = LIBRARY[0]
    return f"{len({p['book'] for p in passages})} books · {len(passages):,} searchable passages"


reload_library()


def ask(question, write=True, use_jev=True):
    passages, index = LIBRARY
    candidates = [passages[i] for i in index.search(question, CANDIDATES)]
    selected = jev_select(question, candidates) if use_jev and candidates else candidates[:TOP_K]
    result = {"passages": selected, "claims": [], "dropped": []}
    if not (write and selected):
        return result
    claims = write_claims(question, selected)
    if not use_jev or not claims:
        result["claims"] = [{"text": c} for c in claims]
        return result
    for claim, p in zip(claims, jev_verify(selected, claims)):
        result["claims" if p >= MIN_SUPPORT else "dropped"].append({"text": claim, "support": round(p, 3)})
    return result


# --- web ---

PAGE = (Path(__file__).parent / "index.html").read_text()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, PAGE.replace("{{stats}}", stats()), "text/html")

    def do_POST(self):
        if self.path == "/upload":
            return self.upload()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            question = str(body.get("question", "")).strip()[:1000]
            if not question:
                return self._send(400, json.dumps({"error": "Ask a question first."}), "application/json")
            out = ask(question, bool(body.get("write", True)), bool(body.get("jev", True)))
            self._send(200, json.dumps(out), "application/json")
        except Exception as e:  # surface API/key errors in the UI instead of a dead request
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")

    def upload(self):
        size = int(self.headers.get("Content-Length", 0))
        if size > MAX_UPLOAD:
            return self._send(413, json.dumps({"error": "File is larger than 100 MB."}), "application/json")
        try:
            title = save_book(unquote(self.headers.get("X-Filename", "")), self.rfile.read(size))
            reload_library()
            self._send(200, json.dumps({"title": title, "stats": stats()}), "application/json")
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}), "application/json")
        except Exception as e:  # e.g. a corrupt or encrypted PDF
            self._send(500, json.dumps({"error": f"Couldn't read that file ({type(e).__name__}: {e})"}), "application/json")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"{stats()} loaded. Open http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
