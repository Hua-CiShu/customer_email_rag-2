"""Customer Email Automation - Streamlit UI.

A Python/Streamlit rebuild of the Dify RAG workflow:
  User Input -> Question Classifier -> Knowledge Retrieval -> LLM -> Output

Run:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

import rag

st.set_page_config(page_title="Customer Email Automation", page_icon="🤖",
                   layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: model & retrieval settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    # API key is entered on the page by each user (never hard-coded, never
    # written to disk). Everything else has sensible defaults baked in.
    api_key = st.text_input(
        "API Key *",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        placeholder="Paste your HKUST GenAI key",
        help="Enter it in the page, no need to configure environment variables in the terminal",
    )
    base_url = st.text_input("Base URL", value=rag.DEFAULT_BASE_URL)
    chat_model = st.text_input("Chat model", value=rag.DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embedding model", value=rag.DEFAULT_EMBED_MODEL)
    top_k = st.slider("Top-K retrieved chunks", 1, 8, 4)
    st.divider()
    st.caption("Knowledge bases")
    for cid, info in rag.CLASS_INFO.items():
        st.markdown(f"**Class {cid}** — {info['name']}  \n"
                    f"`knowledge/{info['kb_file']}`")
    if st.button("🔄 Rebuild embedding index"):
        rag.EMBED_CACHE.unlink(missing_ok=True)
        st.cache_resource.clear()
        st.success("Index cache cleared — will re-embed on next run.")

# ---------------------------------------------------------------------------
# Main panel: inputs (Dify "User Input" node)
# ---------------------------------------------------------------------------
st.title("🤖 Customer Email Automation")
st.caption("RAG Customer Email Automation")

col_in, col_out = st.columns(2)

with col_in:
    st.subheader("📩 Customer Email")
    customer_name = st.text_input("Customer name", placeholder="Megan")
    subject = st.text_input("Subject", placeholder="Order hasn't arrived")
    email_body = st.text_area(
        "Email body *",
        height=220,
        placeholder="Hi, my order was supposed to arrive last week but the "
                    "tracking hasn't updated...",
    )
    run = st.button("✉️ Generate Reply", type="primary",
                    disabled=not email_body.strip())

# ---------------------------------------------------------------------------
# Pipeline execution + outputs (Dify "Output" node)
# ---------------------------------------------------------------------------
with col_out:
    st.subheader("🤖 Generated Reply")
    if not api_key.strip():
        st.warning("Please enter it in the left sidebar first API Key。")
    elif run:
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                st.write("**1/3** Classifying email issue…")
                client = rag.get_client(api_key=api_key.strip(),
                                        base_url=base_url.strip())
                class_id = rag.classify_email(client, subject, email_body, chat_model)
                st.write(f"→ Class {class_id}: {rag.CLASS_INFO[class_id]['name']}")

                st.write("**2/3** Retrieving knowledge…")
                index = rag.build_index(client, embed_model)
                chunks = rag.retrieve(client, index, class_id,
                                      f"{subject}\n{email_body}", top_k, embed_model)
                st.write(f"→ {len(chunks)} chunks from "
                         f"`{rag.CLASS_INFO[class_id]['kb_file']}`")

                st.write("**3/3** Generating reply…")
                reply = rag.generate_reply(client, customer_name, email_body,
                                           chunks, chat_model)
                status.update(label="Done ✅", state="complete")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

        st.markdown(reply)

        with st.expander("🔍 Retrieved knowledge chunks"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}** · score `{c['score']:.3f}`")
                st.markdown(f"> {c['text']}")

        st.download_button("⬇️ Download reply (.txt)", reply,
                           file_name="reply.txt")

st.divider()
st.caption(
    "Workflow mapping — User Input → Form ｜ Question Classifier → `classify_email()` ｜ "
    "Knowledge Retrieval → `build_index()` + `retrieve()` ｜ LLM → `generate_reply()` ｜ "
    "Output → Reply panel on the right"
)
