
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="TNAU Agricultural Schemes AI",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 TNAU Agricultural Schemes AI")
st.caption("Ask questions about Tamil Nadu agricultural government schemes.")

st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload agricultural scheme PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Upload PDFs", type="primary"):
    if uploaded_files:
        files = [
            ("files", (file.name, file.getvalue(), "application/pdf"))
            for file in uploaded_files
        ]

        try:
            with st.spinner("Processing PDF documents..."):
                response = requests.post(
                    f"{API_URL}/upload-pdf/",
                    files=files,
                    timeout=300
                )

            if response.ok:
                result = response.json()
                st.sidebar.success(result["message"])
                st.sidebar.write("Files:", result["files_uploaded"])
                st.sidebar.write("Chunks:", result["chunks_created"])
            else:
                st.sidebar.error(response.text)

        except requests.RequestException as e:
            st.sidebar.error(f"Backend connection error: {e}")
    else:
        st.sidebar.warning("Please select PDF files first.")

st.divider()

st.subheader("💬 Ask Your Agricultural Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What subsidies are available for farmers?"
)

if st.button("Get Answer", type="primary"):
    if question.strip():
        try:
            with st.spinner("Searching documents and generating answer..."):
                response = requests.post(
                    f"{API_URL}/query/",
                    json={"question": question},
                    timeout=120
                )

            if response.ok:
                result = response.json()

                st.subheader("🤖 Answer")
                st.write(result["answer"])

                st.subheader("📚 Sources")

                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i}: {source['file']}"):
                        st.write("Page:", source["page"])
                        st.write(source["content"])

            else:
                st.error(response.text)

        except requests.RequestException as e:
            st.error(f"Backend connection error: {e}")
    else:
        st.warning("Please enter a question.")
