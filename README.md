import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI(title="TNAU Agri-Scheme RAG Backend", version="1.0")

UPLOAD_DIR = "temp_uploads"
DB_FAISS_PATH = "vectorstore/db_faiss"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Pipeline 1: Load PDF -> Split -> Embed -> Save to FAISS Vector DB"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Load PDF Document
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # 3. Create Embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 4. Save/Update FAISS Vector Database
        if os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embeddings)
            
        db.save_local(DB_FAISS_PATH)
        
        return {
            "filename": file.filename, 
            "status": "success",
            "message": f"Successfully processed {len(chunks)} chunks and updated FAISS vector store!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_vector_db(request: QueryRequest):
    """Pipeline 2: Fetch from FAISS, prompt LLM, and parse output with StrOutputParser"""
    if not os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
        raise HTTPException(status_code=400, detail="FAISS Vector DB not found. Please upload a PDF first.")
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Setup Retriever
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        # Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert agricultural assistant specializing in TNAU government schemes (https://agritech.tnau.ac.in). Answer accurately using ONLY the provided context."),
            ("system", "Context:\n{context}"),
            ("user", "{input}")
        ])
        
        # Build chain combining Retrieval, Document Stuffing, and StrOutputParser
        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, document_chain)
        
        response = chain.invoke({"input": request.question})
        
        # Optional explicit string parse representation
        clean_answer = StrOutputParser().parse(response["answer"])
        
        return {
            "answer": clean_answer,
            "source_snippets": [doc.page_content[:250] for doc in response.get("context", [])]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import streamlit as st
import requests

st.set_page_config(page_title="TNAU Agri-Scheme Assistant", page_icon="🌾", layout="centered")

st.title("🌾 TNAU Agricultural Schemes Assistant")
st.caption("Powered by FastAPI, LangChain, FAISS, and Streamlit")

BACKEND_URL = "http://127.0.0.1:8000"

tab1, tab2 = st.tabs(["📂 1. Upload Scheme PDF", "💬 2. Query Schemes"])

with tab1:
    st.header("Upload Agricultural Scheme Document")
    st.write("Upload PDF files downloaded from the [TNAU AgriTech Portal](https://agritech.tnau.ac.in/agricultural_engineering/agriengg_govt_schemes.html).")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process & Store in FAISS"):
            with st.spinner("Uploading and indexing into FAISS vector store..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload-pdf/", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["message"])
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {e}")

with tab2:
    st.header("Search Agricultural Schemes Database")
    
    query = st.text_input("Ask a question about government subsidies, machinery, or engineering schemes:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching FAISS Vector DB and generating response..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/query/", json={"question": query})
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.subheader("💡 Answer:")
                        st.markdown(data["answer"])
                        
                        with st.expander("🔍 View Retrieved Source Snippets"):
                            for i, snippet in enumerate(data["source_snippets"]):
                                st.markdown(f"**Snippet {i+1}:** {snippet}...")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {e}")
        else:
            st.warning("Please enter a question first.")
