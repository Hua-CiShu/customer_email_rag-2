"""SupportFlow - AI customer email copilot.

Product-grade Streamlit UI for the RAG workflow rebuilt from Dify:
  User Input -> Question Classifier -> Knowledge Retrieval -> LLM -> Output

Run:  streamlit run app.py
"""

import html
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import rag

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SupportFlow · AI Email Copilot",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] { background: #F5F6F8; }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }

/* keep the sidebar toggle, hide the rest of the chrome */
#MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- brand bar ---------- */
.brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.brand-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #1D4ED8 0%, #0EA5E9 100%);
    color: #fff; font-weight: 800; font-size: 17px;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: .5px;
}
.brand-name { font-size: 22px; font-weight: 700; color: #0F172A; letter-spacing: -.3px; }
.brand-sub  { font-size: 13px; color: #64748B; margin-top: 1px; }
.brand-right { margin-left: auto; font-size: 12px; color: #94A3B8; text-align: right; }

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E7E9EE !important;
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
.card-title { font-size: 15px; font-weight: 600; color: #0F172A; margin-bottom: 2px; }
.card-hint  { font-size: 12.5px; color: #94A3B8; margin-bottom: 12px; }

/* ---------- inputs & buttons ---------- */
.stTextInput input, .stTextArea textarea {
    border-radius: 10px; border-color: #E2E5EA; font-size: 14px;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #1D4ED8; box-shadow: 0 0 0 1px #1D4ED8;
}
.stTextArea textarea { line-height: 1.6; }

button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: #1D4ED8; border: none; border-radius: 10px;
    font-weight: 600; height: 2.9rem; letter-spacing: .1px;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: #1E40AF;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    border-radius: 10px;
}

/* ---------- result elements ---------- */
.chip {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 600; letter-spacing: .2px;
}
.chip-blue  { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.chip-amber { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }

.reply-card {
    background: #FFFFFF; border: 1px solid #E7E9EE; border-radius: 14px;
    padding: 26px 28px; white-space: pre-wrap; line-height: 1.75;
    font-size: 14.5px; color: #1F2937;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
.meta-line { font-size: 12.5px; color: #94A3B8; margin-top: 10px; }

.empty-card {
    border: 1.5px dashed #D6DAE1; border-radius: 14px; background: #FAFBFC;
    min-height: 420px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
    color: #9AA3B2; text-align: center; padding: 40px;
}
.empty-title { font-size: 14.5px; font-weight: 600; color: #6B7280; }
.empty-sub   { font-size: 12.5px; }

/* ---------- dark sidebar ---------- */
section[data-testid="stSidebar"] { background: #0C1425; border-right: none; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #7C8AA5 !important; }
section[data-testid="stSidebar"] hr { border-color: #1E2A44; }
.side-brand { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -.2px; }
.side-sub   { font-size: 11.5px; color: #7C8AA5; margin-bottom: 6px; }

.footer {
    margin-top: 34px; text-align: center;
    font-size: 11.5px; color: #9AA3B2;
}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — workspace settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand">SupportFlow</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Workspace settings</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "API key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        placeholder="Paste your HKUST GenAI key",
        help="Stored only in this browser session — never written to disk.",
    )

    with st.expander("Advanced"):
        base_url = st.text_input("Base URL", value=rag.DEFAULT_BASE_URL)
        chat_model = st.text_input("Chat model", value=rag.DEFAULT_CHAT_MODEL)
        embed_model = st.text_input("Embedding model", value=rag.DEFAULT_EMBED_MODEL)
        top_k = st.slider("Retrieval depth (top-K)", 1, 8, 4)

    st.divider()
    st.markdown("**Knowledge base**")
    for cid, info in rag.CLASS_INFO.items():
        st.caption(f"Class {cid} · {info['name']}")
        st.caption(f"`knowledge/{info['kb_file']}`")
    if st.button("Rebuild embedding index", use_container_width=True):
        rag.EMBED_CACHE.unlink(missing_ok=True)
        st.cache_resource.clear()
        st.success("Index cache cleared.")

    st.divider()
    st.caption("SupportFlow v1.0 · Internal tool")

# ---------------------------------------------------------------------------
# Brand header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="brand">'
    '<div class="brand-mark">SF</div>'
    '<div><div class="brand-name">SupportFlow</div>'
    '<div class="brand-sub">AI email copilot — every reply grounded in the Acme knowledge base</div></div>'
    '<div class="brand-right">Acme Support Team<br>Internal workspace</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Sample emails
# ---------------------------------------------------------------------------
SAMPLES = {
    "Delayed order (shipping)": {
        "customer_name": "Megan",
        "subject": "Order hasn't arrived",
        "email_body": (
            "Hi, my order was supposed to arrive last week, but the tracking "
            "hasn't updated in five days. Can you tell me where my package is?"
        ),
    },
    "Wrong item received (refund)": {
        "customer_name": "Marcus",
        "subject": "Received the wrong product",
        "email_body": (
            "Hello, I ordered the SmartDesk Pro 2, but when the package arrived "
            "yesterday it contained a chair instead of the desk. I'd like to "
            "return it and get the correct product. What should I do?"
        ),
    },
}


def _load_sample():
    sample = SAMPLES.get(st.session_state.get("sample_choice", ""), None)
    if sample:
        for key, val in sample.items():
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col_in, col_out = st.columns([5, 6], gap="large")

with col_in:
    with st.container(border=True):
        st.markdown('<div class="card-title">New customer email</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card-hint">Paste an incoming message, or load a sample to try the copilot.</div>',
                    unsafe_allow_html=True)

        st.selectbox(
            "Sample emails",
            [""] + list(SAMPLES.keys()),
            key="sample_choice",
            on_change=_load_sample,
            label_visibility="collapsed",
            placeholder="Load a sample email…",
        )

        name_col, subj_col = st.columns([1, 2])
        customer_name = name_col.text_input("Customer name", key="customer_name",
                                            placeholder="Megan")
        subject = subj_col.text_input("Subject", key="subject",
                                      placeholder="Order hasn't arrived")
        email_body = st.text_area(
            "Message",
            key="email_body",
            height=200,
            placeholder="Paste the customer's email here…",
            label_visibility="collapsed",
        )
        run = st.button("Generate draft reply", type="primary",
                        use_container_width=True,
                        disabled=not email_body.strip())

with col_out:
    if run and not api_key.strip():
        st.warning("Add your API key in the sidebar to generate a reply.")
    elif run:
        with st.status("Working on it…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                client = rag.get_client(api_key=api_key.strip(),
                                        base_url=base_url.strip())

                st.write("Classifying request…")
                class_id = rag.classify_email(client, subject, email_body,
                                              chat_model)

                st.write("Searching the knowledge base…")
                index = rag.build_index(client, embed_model)
                chunks = rag.retrieve(client, index, class_id,
                                      f"{subject}\n{email_body}",
                                      top_k, embed_model)

                st.write("Drafting reply…")
                reply = rag.generate_reply(client, customer_name, email_body,
                                           chunks, chat_model)

                st.session_state["result"] = {
                    "class_id": class_id,
                    "class_name": rag.CLASS_INFO[class_id]["name"],
                    "kb_file": rag.CLASS_INFO[class_id]["kb_file"],
                    "chunks": chunks,
                    "reply": reply,
                    "elapsed": time.perf_counter() - t0,
                }
                status.update(label="Draft ready", state="complete",
                              expanded=False)
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error(str(e))

    result = st.session_state.get("result")

    if not result:
        st.markdown(
            '<div class="empty-card">'
            '<svg width="42" height="42" viewBox="0 0 24 24" fill="none" '
            'stroke="#B6BECF" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" '
            'rx="3"/><path d="m2 7 10 6L22 7"/></svg>'
            '<div class="empty-title">Your drafted reply will appear here</div>'
            '<div class="empty-sub">Classified, grounded in company policy, '
            "and ready for review.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        chip_class = "chip-blue" if result["class_id"] == 1 else "chip-amber"
        head_l, head_r = st.columns([3, 2])
        with head_l:
            st.markdown('<div class="card-title">Draft reply</div>',
                        unsafe_allow_html=True)
        with head_r:
            st.markdown(
                f'<div style="text-align:right"><span class="chip {chip_class}">'
                + html.escape(result["class_name"]) + "</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="reply-card">{html.escape(result["reply"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="meta-line">Grounded in '
            f'<b>{html.escape(result["kb_file"])}</b> · '
            f'{len(result["chunks"])} source passages · '
            f'{result["elapsed"]:.1f}s · Always review before sending</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"Grounding sources ({len(result['chunks'])})"):
            for i, c in enumerate(result["chunks"], 1):
                st.markdown(f"**Passage {i}** · relevance `{c['score']:.3f}`")
                st.markdown(f"> {c['text']}")

        st.download_button("Download draft (.txt)", result["reply"],
                           file_name="draft_reply.txt")

st.markdown(
    '<div class="footer">SupportFlow · Retrieval-augmented drafts, '
    "human-approved sends · Acme Support internal tool</div>",
    unsafe_allow_html=True,
)