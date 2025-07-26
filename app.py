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

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure LLM & Embeddings
llm = Gemini(api_key=api_key, model="models/gemini-1.5-flash")
embed_model = GeminiEmbedding(api_key=api_key)
Settings.llm = llm
Settings.embed_model = embed_model

# Set Streamlit Page
st.set_page_config(page_title="📚 Gemini RAG Chatbot", layout="wide")

# --- Session states ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

# --- Sidebar ---
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

# --- Tabs ---
tabs = st.tabs(["📄 Docs Preview","💬 Chat",  "⚙️ Settings"])

# --- Docs Preview Tab ---
with tabs[0]:
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
                    st.code(text[:1000])  # Preview first 1000 chars
                if page_num >= 3:
                    st.info("Preview limited to 3 pages.")
                    break
        else:
            st.markdown(f"📄 *{filename} (preview not supported yet)*")

# --- Chat Tab ---
with tabs[1]:
    st.subheader("💬 Ask Questions")
    if st.session_state.query_engine:
        prompt = st.chat_input("Ask something about your documents...")
        if prompt:
            with st.spinner("Gemini is thinking..."):
                response = st.session_state.query_engine.query(prompt)
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": prompt,
                    "time": timestamp
                })
                st.session_state.chat_history.append({
                    "role": "gemini",
                    "content": response.response,
                    "time": timestamp
                })

    # Display chat
theme = st.get_option("theme.base")
user_color = "#f0f0f0" if theme == "light" else "#394648"
bot_color = "#e2e8f0" if theme == "light" else "#453c2f"
text_color = "#000000" if theme == "light" else "#ffffff"

for msg in st.session_state.chat_history:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    bg_color = user_color if msg["role"] == "user" else bot_color

    st.markdown(f"""
    <div style='
        background-color: {bg_color};
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: {text_color};
    '>
    <b>{avatar} {msg['role'].capitalize()} [{msg['time']}]:</b><br>{msg["content"]}
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.query_engine:
        st.info("⚠️ Please upload files and click 'Rebuild Index' in the sidebar.")

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
