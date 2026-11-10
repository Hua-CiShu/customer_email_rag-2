# SupportFlow · AI Customer Email Copilot

SupportFlow turns incoming customer emails into grounded, ready-to-review reply
drafts. Paste a full email or drop a file — it parses the sender, subject and
body, classifies the issue, retrieves the relevant company policy, and drafts a
reply that cites real knowledge-base content instead of hallucinating.

Originally prototyped as a Dify workflow ("Customer Email Automation"), rebuilt
as a standalone product with Python + Streamlit.

```
Parse email → Classify issue → Retrieve policy → Draft reply
```

## Features

- **Zero-friction intake** — paste a raw email (headers included) or upload a
  `.txt` / `.md` / `.docx` file; an LLM extracts sender, subject and body.
  Manual field entry is still available as a fallback.
- **Issue classification** — routes each email to *Shipping / Delivery* or
  *Refund / Return / Exchange* with an LLM classifier.
- **Grounded retrieval (RAG)** — each category has its own knowledge base;
  top-K passages are retrieved via embeddings with pure NumPy cosine
  similarity (no vector database required).
- **Transparent drafting** — every draft shows its category, source passages,
  grounding file and generation time, with an "always review before sending"
  reminder.
- **Privacy-friendly keys** — each user pastes their own API key in the
  sidebar; it lives only in the browser session and is never written to disk.
- **Fast restarts** — embeddings are cached to `.embed_cache.npz` and rebuilt
  on demand from the sidebar.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the page, paste your API key in the sidebar, load a sample email (or
paste your own), and hit **Analyze & draft reply**.

The default endpoint is the HKUST GenAI gateway; the base URL and both models
can be changed under **Advanced** in the sidebar, or via `.env`
(see `.env.example`) for any OpenAI-compatible provider.

## Deploy as a public web app

1. Push this repo to GitHub (`.gitignore` already excludes `.env` and caches —
   never commit an API key).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **Create app** → pick the repo → main file path `app.py` → **Deploy**.
4. Share the generated `*.streamlit.app` link; visitors each enter their own
   API key.

Every push to `main` redeploys automatically.

## Project structure

| Path | What it does |
|---|---|
| `app.py` | Product UI: intake tabs (paste / file / manual), pipeline status, result cards, grounding sources |
| `rag.py` | Core logic: email parsing, classification, chunking, embeddings, retrieval, reply drafting |
| `knowledge/shipping_policy.md` | Knowledge base for Class 1 — replace with your own document |
| `knowledge/refund_policy.md` | Knowledge base for Class 2 — replace with your own document |
| `.env.example` | Optional environment-variable template |

## Customizing the knowledge base

Replace the two markdown files under `knowledge/` (or add new categories in
`CLASS_INFO` in `rag.py`), then click **Rebuild embedding index** in the
sidebar to re-embed.

## Tech notes

- Python + Streamlit, OpenAI-compatible chat + embeddings
- Retrieval: chunked markdown → embeddings → cosine similarity (NumPy)
- Word file parsing via `python-docx`