# Reading Library

Ask questions about your own books and get answers backed by the passages they came from. Claims the books don't support are removed before you see the answer.

A chat model (through [LiteLLM](https://github.com/BerriAI/litellm)) writes the answer. [Jev](https://typesafe.ai), TypeSafe's System One model, does two checks:

1. **Picks the passages** that actually help answer the question.
2. **Verifies each claim** in the answer against those passages and removes the unsupported ones.

## How it works

```
question
  → BM25 keyword search      top 30 passages, runs locally, free
  → Jev: relevant?           keep passages with p(yes) ≥ 0.5, max 6
  → LiteLLM writes claims    one per line, each cited like [1]
  → Jev: supported?          keep claims with p(yes) ≥ 0.7
  → answer + sources + removed claims
```

Jev doesn't generate text. It answers yes/no questions with probabilities, all in one request, so checking 30 passages or 8 claims costs one API call each.

## Setup

Requires Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```sh
TYPESAFE_API_KEY=...          # https://console.typesafe.ai

# the model that writes answers: any LiteLLM model + its key
LLM_MODEL=openai/gpt-4o
OPENAI_API_KEY=...
```

Other options: `anthropic/claude-sonnet-5` with `ANTHROPIC_API_KEY`, or `ollama/llama3` for a local model with no key.

## Run

```sh
.venv/bin/python app.py
```

Open http://localhost:8000.

## Adding books

- **In the app:** click **Add books** or drag PDF/TXT files onto the page. They're searchable right away.
- **By hand:** put `.txt` files in `books/` and restart the app.

The file name becomes the book title, e.g. `Deep Work - Cal Newport.pdf`. Scanned PDFs (page images with no text layer) aren't supported. Books are kept out of git by `.gitignore`.

## Options in the page

| Switch | Off means |
|---|---|
| Write an answer | Show the passages only, with no AI-written answer |
| Use Jev checks | Use plain keyword results and an unchecked answer |

Turning Jev off lets you compare answers with and without the checks.

## Tuning

Constants at the top of `app.py`:

| Name | Default | What it does |
|---|---|---|
| `CANDIDATES` | 30 | Passages from keyword search sent to Jev |
| `TOP_K` | 6 | Most passages given to the writer |
| `MIN_RELEVANCE` | 0.5 | Jev score needed to keep a passage |
| `MIN_SUPPORT` | 0.7 | Jev score needed to keep a claim |

## Tests

```sh
.venv/bin/python test_app.py
```

Runs offline. Jev and the LLM are stubbed, so no API keys are needed.

## Privacy

- With Jev on, your question, the selected passages and the draft claims are sent to TypeSafe.
- With "Write an answer" on, the selected passages are sent to your LLM provider. Use an `ollama/...` model to keep that step on your machine.
- Book files and the search index stay local.

## Limitations

- Search matches words, not meaning, so a question about "focus" may miss a passage about "concentration".
- Each claim is checked against all selected passages, not only the ones it cites.
- Jev's judgements are probabilities and can be wrong. Adjust the thresholds for your books.
- The server listens on `127.0.0.1` only and has no login. It's meant for personal use.

## Files

```
app.py          search, Jev calls, LiteLLM call, web server
index.html      the web page
test_app.py     offline tests
books/          your .txt books (created from uploads too)
```
