"""Offline check: Jev and LiteLLM are stubbed. Run: .venv/bin/python test_app.py"""
import app

import tempfile
from pathlib import Path

PASSAGES = [
    {"book": "Deep Work", "text": "Schedule deep work blocks every morning and protect them from email."},
    {"book": "Atomic Habits", "text": "Habit stacking ties a new habit to an existing routine."},
    {"book": "Cooking", "text": "Salt the pasta water generously."},
]
app.LIBRARY = (PASSAGES, app.BM25(PASSAGES))

assert app.LIBRARY[1].search("deep work routine", 5)[0] == 0
assert app.parse_claims("Intro\n- A [1]\n* B [2]\nnope") == ["A [1]", "B [2]"]

app.jev = lambda state, qs: {k: (0.9 if k in ("p0", "p1", "c0") else 0.1) for k in qs}
app.write_claims = lambda q, ps: ["Block mornings for deep work [1]", "Deep work cures cancer [1]"]
r = app.ask("deep work routine")
assert [p["book"] for p in r["passages"]] == ["Deep Work", "Atomic Habits"], r
assert [c["text"] for c in r["claims"]] == ["Block mornings for deep work [1]"], r
assert [c["text"] for c in r["dropped"]] == ["Deep work cures cancer [1]"], r
assert app.ask("deep work", write=False)["claims"] == []
assert len(app.ask("deep work", use_jev=False)["claims"]) == 2
# uploads: long text with no blank lines (typical PDF output) must still split into ~250-word passages
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    assert app.save_book("../My Book?.txt", ("word " * 600).encode(), d) == "My Book"
    assert [len(p["text"].split()) for p in app.load_passages(d)] == [250, 250, 100]
    for bad in ["notes.docx", "empty.txt"]:
        try:
            app.save_book(bad, b"too short", d)
            raise AssertionError(bad)
        except ValueError:
            pass
print("ok")
