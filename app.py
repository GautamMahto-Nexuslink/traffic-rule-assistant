import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="TN Traffic Rules Assistant",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styles ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .session-btn button { text-align: left !important; }
        .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.database import init_db, create_session, get_user_sessions, add_message, get_messages, update_session_title, delete_session
from src.auth import register, login

init_db()

BASE_DIR = Path(__file__).parent
EMBEDDINGS_DIR = str(BASE_DIR / "data" / "embeddings")
CHUNKS_DIR = str(BASE_DIR / "data" / "chunks")
PIPELINE_READY = (
    (BASE_DIR / "data" / "embeddings" / "faiss_index.bin").exists()
    and (BASE_DIR / "data" / "chunks").exists()
)


# ── Cached RAG resources (loaded once for all users) ─────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_rag():
    from src.chatbot import load_resources
    return load_resources(EMBEDDINGS_DIR, CHUNKS_DIR)


@st.cache_resource
def get_groq_client():
    from groq import Groq
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "user_id": None,
    "username": None,
    "session_id": None,
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def logout():
    for key in ("user_id", "username", "session_id", "messages"):
        st.session_state[key] = None if key != "messages" else []


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.user_id is None:
    col_l, col_m, col_r = st.columns([1, 1.8, 1])
    with col_m:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🚦 TN Traffic Rules Assistant")
        st.caption("Ask anything about Tamil Nadu traffic rules")
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    ok, user_id, msg = login(username, password)
                    if ok:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_signup:
            with st.form("signup_form", clear_on_submit=True):
                new_username = st.text_input("Choose a username", placeholder="At least 3 characters")
                new_password = st.text_input("Choose a password", type="password", placeholder="At least 6 characters")
                confirm_pw = st.text_input("Confirm password", type="password", placeholder="Repeat password")
                submitted_signup = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submitted_signup:
                if not new_username or not new_password or not confirm_pw:
                    st.error("Please fill in all fields.")
                elif new_password != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register(new_username, new_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()

        st.divider()

        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

        st.markdown("**Previous Chats**")
        sessions = get_user_sessions(st.session_state.user_id)

        if not sessions:
            st.caption("No chats yet. Start a new one!")
        else:
            for s in sessions:
                col_title, col_del = st.columns([5, 1])
                label = s["title"] if len(s["title"]) <= 28 else s["title"][:26] + "…"
                is_active = st.session_state.session_id == s["id"]

                with col_title:
                    if st.button(
                        ("▶ " if is_active else "") + label,
                        key=f"sess_{s['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.session_id = s["id"]
                        st.session_state.messages = [
                            dict(m) for m in get_messages(s["id"])
                        ]
                        st.rerun()

                with col_del:
                    if st.button("🗑", key=f"del_{s['id']}"):
                        delete_session(s["id"], st.session_state.user_id)
                        if st.session_state.session_id == s["id"]:
                            st.session_state.session_id = None
                            st.session_state.messages = []
                        st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown("## 🚦 TN Traffic Rules Assistant")

    if not PIPELINE_READY:
        st.warning(
            "Knowledge base not found. Run `python main.py pipeline` first, then refresh this page.",
            icon="⚠️",
        )
        st.stop()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about TN traffic rules…"):
        # Create a new session on first message
        if st.session_state.session_id is None:
            st.session_state.session_id = create_session(st.session_state.user_id)

        # Show & persist user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        add_message(st.session_state.session_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate & show assistant reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from src.chatbot import retrieve, ask_groq
                index, chunks, model = load_rag()
                client = get_groq_client()
                relevant = retrieve(prompt, index, chunks, model)
                answer = ask_groq(prompt, relevant, client)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        add_message(st.session_state.session_id, "assistant", answer)

        # Auto-title the session from the first user message
        sessions = get_user_sessions(st.session_state.user_id)
        current = next((s for s in sessions if s["id"] == st.session_state.session_id), None)
        if current and current["title"] == "New Chat":
            title = prompt[:45] + ("…" if len(prompt) > 45 else "")
            update_session_title(st.session_state.session_id, title)
