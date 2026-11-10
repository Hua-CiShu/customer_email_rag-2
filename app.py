"""SupportFlow - AI customer email copilot.

Product-grade Streamlit UI for the RAG workflow rebuilt from Dify:
  Parse -> Question Classifier -> Knowledge Retrieval -> LLM -> Draft

Run:  streamlit run app.py
"""

import html
import io
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
.block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 2rem; }

#MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- brand bar ---------- */
.brand { display: flex; align-items: center; gap: 14px; }
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

/* ---------- pipeline strip ---------- */
.flow {
    display: flex; align-items: center; gap: 12px;
    background: #FFFFFF; border: 1px solid #E7E9EE; border-radius: 12px;
    padding: 12px 18px; margin: 14px 0 18px;
}
.flow-step { display: flex; align-items: center; gap: 8px;
    font-size: 12.5px; font-weight: 600; color: #334155; }
.flow-num {
    width: 20px; height: 20px; border-radius: 6px;
    background: #EFF6FF; color: #1D4ED8; font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.flow-arrow { color: #CBD5E1; font-size: 13px; }
.flow-note { margin-left: auto; font-size: 11.5px; color: #94A3B8; }

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E7E9EE !important;
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
.card-title { font-size: 15px; font-weight: 600; color: #0F172A; margin-bottom: 2px; }
.card-hint  { font-size: 12.5px; color: #94A3B8; margin-bottom: 10px; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #EEF0F3; }
.stTabs [data-baseweb="tab"] {
    font-size: 13px; font-weight: 600; color: #64748B;
    padding: 6px 14px;
}
.stTabs [aria-selected="true"] { color: #1D4ED8 !important; }

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

.stat-row { display: flex; gap: 12px; margin: 12px 0 14px; }
.stat {
    flex: 1; background: #F8FAFC; border: 1px solid #E7E9EE;
    border-radius: 12px; padding: 12px 16px;
}
.stat-label {
    font-size: 10.5px; color: #94A3B8; text-transform: uppercase;
    letter-spacing: .6px; font-weight: 600; margin-bottom: 5px;
}
.stat-value { font-size: 15px; font-weight: 700; color: #0F172A; }

.reply-card {
    background: #FFFFFF; border: 1px solid #E7E9EE; border-radius: 14px;
    padding: 24px 26px; white-space: pre-wrap; line-height: 1.75;
    font-size: 14.5px; color: #1F2937;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
.meta-line { font-size: 12.5px; color: #94A3B8; margin-top: 10px; }

.empty-card {
    border: 1.5px dashed #D6DAE1; border-radius: 14px; background: #FAFBFC;
    min-height: 360px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
    color: #9AA3B2; text-align: center; padding: 36px;
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
    margin-top: 26px; text-align: center;
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
    st.caption("SupportFlow v1.1 · Internal tool")

# ---------------------------------------------------------------------------
# Brand header + pipeline strip
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
st.markdown(
    '<div class="flow">'
    '<div class="flow-step"><span class="flow-num">1</span>Parse email</div>'
    '<span class="flow-arrow">→</span>'
    '<div class="flow-step"><span class="flow-num">2</span>Classify issue</div>'
    '<span class="flow-arrow">→</span>'
    '<div class="flow-step"><span class="flow-num">3</span>Retrieve policy</div>'
    '<span class="flow-arrow">→</span>'
    '<div class="flow-step"><span class="flow-num">4</span>Draft reply</div>'
    '<span class="flow-note">RAG pipeline · paste a full email or drop a file</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sample emails (manual tab)
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


def _read_uploaded(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".docx"):
        import docx  # python-docx
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError("Unsupported file type — please use .txt or .docx")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col_in, col_out = st.columns([5, 6], gap="large")

with col_in:
    with st.container(border=True):
        st.markdown('<div class="card-title">Incoming email</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card-hint">Paste the full email or drop a file — SupportFlow extracts the sender, subject and body for you.</div>',
                    unsafe_allow_html=True)

        tab_paste, tab_file, tab_manual = st.tabs(
            ["Full email", "File upload", "Manual entry"])

        with tab_paste:
            raw_input = st.text_area(
                "Raw email",
                key="raw_email",
                height=270,
                placeholder=(
                    "Paste the complete email here, headers included. For example:\n\n"
                    "From: Megan <megan@example.com>\n"
                    "Subject: Order hasn't arrived\n\n"
                    "Hi, my order was supposed to arrive last week…"
                ),
                label_visibility="collapsed",
            )

        with tab_file:
            uploaded = st.file_uploader(
                "Upload email file",
                type=["txt", "md", "docx"],
                key="email_file",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                st.caption(f"{uploaded.name} · {uploaded.size / 1024:.1f} KB — "
                           "will be parsed automatically")
            st.caption("Supported formats: .txt, .md, .docx")

        with tab_manual:
            st.selectbox(
                "Sample emails",
                [""] + list(SAMPLES.keys()),
                key="sample_choice",
                on_change=_load_sample,
                label_visibility="collapsed",
                placeholder="Load a sample email…",
            )
            name_col, subj_col = st.columns([1, 2])
            customer_name = name_col.text_input("Customer name",
                                                key="customer_name",
                                                placeholder="Megan")
            subject = subj_col.text_input("Subject", key="subject",
                                          placeholder="Order hasn't arrived")
            email_body = st.text_area(
                "Message",
                key="email_body",
                height=140,
                placeholder="Paste the customer's email here…",
                label_visibility="collapsed",
            )

        has_input = bool(
            (raw_input or "").strip()
            or st.session_state.get("email_file") is not None
            or (st.session_state.get("email_body") or "").strip()
        )
        run = st.button("Analyze & draft reply", type="primary",
                        use_container_width=True, disabled=not has_input)

with col_out:
    if run and not api_key.strip():
        st.warning("Add your API key in the sidebar to generate a reply.")
    elif run:
        with st.status("Working on it…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                client = rag.get_client(api_key=api_key.strip(),
                                        base_url=base_url.strip())

                # --- 1. resolve input: file > pasted raw email > manual ---
                uploaded_file = st.session_state.get("email_file")
                if uploaded_file is not None:
                    st.write("Reading file…")
                    raw_text = _read_uploaded(uploaded_file)
                    source_label = uploaded_file.name
                elif (raw_input or "").strip():
                    raw_text = raw_input
                    source_label = "pasted email"
                else:
                    raw_text = ""
                    source_label = "manual entry"

                if raw_text:
                    st.write("Parsing email…")
                    parsed = rag.parse_email(client, raw_text, chat_model)
                    name = parsed["customer_name"]
                    subj = parsed["subject"]
                    body = parsed["body"]
                else:
                    name = st.session_state.get("customer_name", "")
                    subj = st.session_state.get("subject", "")
                    body = st.session_state.get("email_body", "")

                if not body.strip():
                    raise ValueError("No email content found to analyze.")

                # --- 2. classify ---
                st.write("Classifying request…")
                class_id = rag.classify_email(client, subj, body, chat_model)

                # --- 3. retrieve ---
                st.write("Searching the knowledge base…")
                index = rag.build_index(client, embed_model)
                chunks = rag.retrieve(client, index, class_id,
                                      f"{subj}\n{body}", top_k, embed_model)

                # --- 4. draft ---
                st.write("Drafting reply…")
                reply = rag.generate_reply(client, name, body, chunks,
                                           chat_model)

                st.session_state["result"] = {
                    "class_id": class_id,
                    "class_name": rag.CLASS_INFO[class_id]["name"],
                    "kb_file": rag.CLASS_INFO[class_id]["kb_file"],
                    "chunks": chunks,
                    "reply": reply,
                    "from_name": name,
                    "subject": subj,
                    "source_label": source_label,
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
        from_line = html.escape(result["from_name"] or "Unknown sender")
        subj_line = html.escape(result["subject"] or "(no subject)")

        head_l, head_r = st.columns([3, 2])
        with head_l:
            st.markdown('<div class="card-title">Draft reply</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="card-hint" style="margin-bottom:0">'
                f'From <b>{from_line}</b> · {subj_line} · '
                f'{html.escape(result["source_label"])}</div>',
                unsafe_allow_html=True,
            )
        with head_r:
            st.markdown(
                f'<div style="text-align:right;padding-top:6px">'
                f'<span class="chip {chip_class}">'
                + html.escape(result["class_name"]) + "</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="stat-row">'
            f'<div class="stat"><div class="stat-label">Category</div>'
            f'<div class="stat-value">Class {result["class_id"]}</div></div>'
            f'<div class="stat"><div class="stat-label">Source passages</div>'
            f'<div class="stat-value">{len(result["chunks"])}</div></div>'
            f'<div class="stat"><div class="stat-label">Draft time</div>'
            f'<div class="stat-value">{result["elapsed"]:.1f}s</div></div>'
            "</div>",
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
            "Always review before sending</div>",
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