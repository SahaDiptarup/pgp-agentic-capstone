"""
Streamlit UI for the PGP Agentic RAG app.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.ingest import ingest_files  # noqa: E402
from app.agent import run_agent  # noqa: E402

st.set_page_config(
    page_title="Document Q&A",
    page_icon="",
    layout="wide",
)

st.title("Document Q&A Agent")
st.caption(
    "Upload documents (PDF, TXT, CSV, Excel) and ask questions. "
    "Powered by Gemini + RAG."
)

# sidebar for uploading docs
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "csv", "xls", "xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents", type="primary"):
        file_pairs = [(f.name, f.read()) for f in uploaded_files]
        with st.spinner("Indexing documents..."):
            try:
                results = ingest_files(file_pairs)
                for fname, result in results.items():
                    if isinstance(result, int):
                        st.success(f"{fname}: {result} chunks indexed")
                    else:
                        st.error(f"{fname}: {result}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.markdown(
        "**How to use**\n"
        "1. Upload one or more documents above.\n"
        "2. Click **Ingest Documents**.\n"
        "3. Ask a question in the chat below.\n\n"
        "_Make sure GOOGLE_API_KEY is set in your .env file._"
    )

# chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_agent(prompt)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.write(f"- {src}")
            except Exception as exc:  # noqa: BLE001
                answer = f"An error occurred: {exc}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
