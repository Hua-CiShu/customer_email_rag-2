"""Core RAG logic for the Customer Email Automation app.

Rebuild of the Dify workflow "Customer Email Automation":
  User Input -> Question Classifier -> Knowledge Retrieval -> LLM -> Output
"""

from __future__ import annotations  # allow str | None etc. on Python < 3.10

import os
import re
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# Each classifier branch maps to its own knowledge base (like the two
# Knowledge Retrieval nodes in the Dify workflow).
CLASS_INFO = {
    1: {
        "name": "Shipping / Delivery Issues",
        "kb_file": "shipping_policy.md",
        "description": "Late packages, shipping delays, tracking issues, delivery status.",
    },
    2: {
        "name": "Refund / Return / Exchange",
        "kb_file": "refund_policy.md",
        "description": "Refund, return, or exchange requests.",
    },
}

# Defaults are baked in so the app works out of the box — no terminal
# exports needed. Environment variables (or .env) can still override them.
DEFAULT_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://hkust.azure-api.net/hkust-genai/v1"
)
DEFAULT_CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
EMBED_CACHE = Path(__file__).parent / ".embed_cache.npz"


def get_client(api_key: str | None = None,
               base_url: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client.

    - api_key: from the caller (e.g. the Streamlit sidebar) or OPENAI_API_KEY.
      The key is never hard-coded — each user enters their own on the page.
    - base_url: defaults to the HKUST GenAI gateway; overridable per call.
    - OPENAI_EXTRA_HEADERS: optional JSON string of extra request headers.
      Gateways whose URL contains "azure-api.net" (e.g. HKUST GenAI) get the
      {"api-key": <key>} header automatically.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key provided. Enter it in the sidebar, or set OPENAI_API_KEY."
        )
    base_url = base_url or DEFAULT_BASE_URL

    extra_headers = None
    raw = os.environ.get("OPENAI_EXTRA_HEADERS")
    if raw:
        extra_headers = json.loads(raw)
    elif base_url and "azure-api.net" in base_url:
        extra_headers = {"api-key": api_key}

    return OpenAI(api_key=api_key, base_url=base_url,
                  default_headers=extra_headers)


# ---------------------------------------------------------------------------
# 1. Question classifier  (Dify node: "determine email issue")
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """You are an email classifier for Acme Company customer support.

Classify the customer's email into exactly ONE of these classes:

CLASS 1 - Shipping / Delivery Issues
  Emails mentioning late packages, shipping delays, tracking issues, or
  delivery status. Examples: "Hasn't arrived", "tracking number missing",
  "delayed shipment".

CLASS 2 - Refund / Return / Exchange
  Emails asking for a refund, return, or exchange.
  Examples: "want a refund", "return my order", "doesn't fit".

Reply with ONLY the class number: 1 or 2. No other text."""


def classify_email(client: OpenAI, subject: str, body: str,
                   model: str = DEFAULT_CHAT_MODEL) -> int:
    """Return 1 (shipping) or 2 (refund/return/exchange)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
        ],
    )
    text = resp.choices[0].message.content.strip()
    match = re.search(r"[12]", text)
    return int(match.group()) if match else 1


# ---------------------------------------------------------------------------
# 2. Knowledge retrieval  (Dify nodes: "Knowledge Retrieval" 1 & 2)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 450) -> list[str]:
    """Split a markdown knowledge file into chunks on blank lines,
    merging small paragraphs so chunks stay meaningful."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def load_kb_chunks(kb_file: str) -> list[str]:
    return chunk_text((KNOWLEDGE_DIR / kb_file).read_text(encoding="utf-8"))


def _embed(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def build_index(client: OpenAI, embed_model: str = DEFAULT_EMBED_MODEL) -> dict:
    """Embed every knowledge base and return {class_id: (chunks, vectors)}.

    Vectors are L2-normalised so retrieval is a simple dot product.
    A disk cache avoids re-embedding on every restart."""
    if EMBED_CACHE.exists():
        data = np.load(EMBED_CACHE, allow_pickle=True)
        return {
            int(k): (data[f"chunks_{k}"].tolist(), data[f"vec_{k}"])
            for k in ("1", "2")
        }

    index = {}
    for class_id, info in CLASS_INFO.items():
        chunks = load_kb_chunks(info["kb_file"])
        vectors = _embed(client, chunks, embed_model)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index[class_id] = (chunks, vectors)

    np.savez(
        EMBED_CACHE,
        **{f"chunks_{k}": np.array(v[0], dtype=object) for k, v in index.items()},
        **{f"vec_{k}": v[1] for k, v in index.items()},
    )
    return index


def retrieve(client: OpenAI, index: dict, class_id: int, query: str,
             top_k: int = 4, embed_model: str = DEFAULT_EMBED_MODEL) -> list[dict]:
    """Return the top_k most similar chunks from the class's knowledge base."""
    chunks, vectors = index[class_id]
    q = _embed(client, [query], embed_model)[0]
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i], "score": float(scores[i])}
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# 3. Reply generation  (Dify nodes: "LLM" & "LLM 2")
# ---------------------------------------------------------------------------

REPLY_PROMPT = """You are a professional customer support assistant for **Acme Company**, known for clear, friendly, and accurate communication.
Your task is to:
- Read the customer's email.
- Review the company's internal knowledge base context (below).
- Use that information to write a factual, helpful, and natural reply that addresses the customer's request.
---
**Customer email:**
{customer_email_body}
---
**Relevant company knowledge (flattened text):**
{context}
---
### Reply Instructions:
- Greet the customer by {customer_name} if possible (use "there" if no name provided).
- Acknowledge their issue in natural language.
- Use factual details from the knowledge above.
- End with:
"Best,
The Acme Support Team"
Only return the final email body."""


def generate_reply(client: OpenAI, customer_name: str, email_body: str,
                   context_chunks: list[dict],
                   model: str = DEFAULT_CHAT_MODEL) -> str:
    context = "\n\n".join(c["text"] for c in context_chunks)
    prompt = REPLY_PROMPT.format(
        customer_email_body=email_body,
        context=context,
        customer_name=customer_name or "there",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(customer_name: str, subject: str, email_body: str,
                 top_k: int = 4, chat_model: str = DEFAULT_CHAT_MODEL,
                 embed_model: str = DEFAULT_EMBED_MODEL,
                 api_key: str | None = None,
                 base_url: str | None = None) -> dict:
    client = get_client(api_key=api_key, base_url=base_url)
    class_id = classify_email(client, subject, email_body, chat_model)
    index = build_index(client, embed_model)
    query = f"{subject}\n{email_body}"  # retrieval query
    chunks = retrieve(client, index, class_id, query, top_k, embed_model)
    reply = generate_reply(client, customer_name, email_body, chunks, chat_model)
    return {
        "class_id": class_id,
        "class_name": CLASS_INFO[class_id]["name"],
        "retrieved_chunks": chunks,
        "reply": reply,
    }
