import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from utils.loader import load_all_documents
import fitz  # PyMuPDF
import json
import time
import streamlit.components.v1 as components

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini LLM and Embeddings
llm = Gemini(api_key=api_key, model="models/gemini-1.5-flash")
embed_model = GeminiEmbedding(api_key=api_key)
Settings.llm = llm
Settings.embed_model = embed_model

# Streamlit Page Config
st.set_page_config(page_title="📚 DocBot", layout="wide")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

# Sidebar - Upload Documents
with st.sidebar:
    st.header("📁 Upload Documents")
    files = st.file_uploader("Upload PDF, DOCX, TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if files:
        os.makedirs("data", exist_ok=True)
        new_files = []
        for file in files:
            file_path = os.path.join("data", file.name)
            new_files.append(file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        if new_files != st.session_state.uploaded_filenames:
            st.session_state.uploaded_filenames = new_files
            st.session_state.chat_history.clear()
            st.session_state.query_engine = None
            st.warning("🛠️ New files uploaded. Please rebuild index below.")

        if st.button("🔁 Rebuild Index"):
            with st.spinner("Building vector index..."):
                documents = load_all_documents("data")
                index = VectorStoreIndex.from_documents(documents)
                st.session_state.query_engine = index.as_query_engine()
            st.success("✅ Index built! You can now chat →")

    if st.button("🗑️ Clear All & Reset"):
        st.session_state.chat_history.clear()
        st.session_state.query_engine = None
        st.session_state.uploaded_filenames = []
        st.rerun()

# Tabs UI (Chat is default)
tabs = st.tabs(["💬 Chat", "📄 Docs Preview", "⚙️ Settings"])

# --- Chat Tab ---
with tabs[0]:
    st.markdown("<h2 style='color:green;'>💬 Ask Questions</h2>", unsafe_allow_html=True)

    chat_container = st.container()
    scroll_code = """
    <script>
    var chatDiv = window.parent.document.querySelector('section.main');
    if (chatDiv) {
        chatDiv.scrollTop = chatDiv.scrollHeight;
    }
    </script>
    """

    if st.session_state.query_engine:
        prompt = st.chat_input("Ask something about your documents...")
        if prompt:
            user_time = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "time": user_time
            })

            with st.spinner("🤖 DocBot is thinking..."):
                time.sleep(0.5)
                response = st.session_state.query_engine.query(prompt)

            DocBot_time = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                "role": "DocBot",
                "content": response.response,
                "time": DocBot_time
            })

            components.html(scroll_code, height=0)
    else:
        st.warning("⚠️ Please upload documents and click '🔁 Rebuild Index' in the sidebar to enable asking questions.")

    with chat_container:
        theme = st.get_option("theme.base")
        user_bg = "#dcf8c6" if theme == "light" else "#2e3b3c"
        bot_bg = "#f1f0f0" if theme == "light" else "#383232"
        text_color = "#000000" if theme == "light" else "#ffffff"

        for msg in st.session_state.chat_history:
            is_user = msg["role"] == "user"
            avatar_emoji = "🧑" if is_user else "🤖"
            bg_color = user_bg if is_user else bot_bg
            align = "flex-end" if is_user else "flex-start"

            st.markdown(f"""
            <div style='
                display: flex;
                justify-content: {align};
                margin-bottom: 1rem;
                align-items: flex-start;
            '>
              <div style='
                  background-color: {bg_color};
                  padding: 1rem;
                  border-radius: 1rem;
                  max-width: 85%;
                  color: {text_color};
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                  white-space: pre-wrap;
                  overflow-wrap: break-word;
              '>
                <b>{avatar_emoji} {msg["role"].capitalize()} [{msg["time"]}]</b><br>
                {msg["content"]}
              </div>
            </div>
            """, unsafe_allow_html=True)

        components.html(scroll_code, height=0)

# --- Docs Preview Tab ---
with tabs[1]:
    st.subheader("📄 Document Preview")
    for filename in st.session_state.uploaded_filenames:
        if filename.endswith(".pdf"):
            path = os.path.join("data", filename)
            doc = fitz.open(path)
            st.markdown(f"**📘 {filename}**")
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    st.markdown(f"<b>Page {page_num}</b>", unsafe_allow_html=True)
                    st.code(text[:1000])  # Preview first 1000 characters
                if page_num >= 3:
                    st.info("Preview limited to 3 pages.")
                    break
        else:
            st.markdown(f"📄 *{filename} (preview not supported yet)*")

# --- Settings Tab ---
with tabs[2]:
    st.subheader("⚙️ Chat Settings & Export")
    if st.button("💾 Export Chat History (.json)"):
        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button("📥 Download JSON", chat_json, file_name="chat_history.json", mime="application/json")
        else:
            st.warning("No chat history available.")

    if st.button("💾 Export Chat History (.txt)"):
        if st.session_state.chat_history:
            chat_text = "\n\n".join([f"{msg['role'].capitalize()} [{msg['time']}]:\n{msg['content']}" for msg in st.session_state.chat_history])
            st.download_button("📥 Download TXT", chat_text, file_name="chat_history.txt", mime="text/plain")
        else:
            st.warning("No chat history available.")
