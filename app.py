import os

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question


# set the working directory
work_dir = os.getcwd()
docs_dir = os.path.join(work_dir, 'docs')
st.title("🤖 OpenAI Assistant - Documents RAG")

# initialize session variables
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

print(st.session_state)
#FFIle Upload Widget
uploaded_files = st.file_uploader("Upload a PDF file",type=["pdf"],accept_multiple_files=True)

if uploaded_files:

    for uploaded_file in uploaded_files:

        if uploaded_file.name not in st.session_state.processed_files:

            save_path = os.path.join(docs_dir, uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            vectordb = process_document_to_chroma_db(uploaded_file.name)

            st.session_state.vectordb = vectordb
            st.session_state.processed_files.add(uploaded_file.name)

            st.success(f"Processed: {uploaded_file.name}")

user_question = st.text_area("Ask a question about the documents")
if st.button("Answer"):
    if st.session_state.vectordb is None:
        st.warning("Upload a document first")
    else:
        answer, source_docs = answer_question(
            user_question,
            st.session_state.vectordb
        )

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")

        for doc in source_docs:
            st.write(doc.metadata.get("source"))