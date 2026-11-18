
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY missing. Check your .env file.")

# Initialize application
app = FastAPI(title="TNAU Agricultural Schemes RAG API")

# Paths
UPLOAD_DIR = BASE_DIR / "temp_uploads"
FAISS_DIR = BASE_DIR / "faiss_index"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

# Load existing FAISS index if available
vector_store = None

if FAISS_DIR.exists():
    try:
        vector_store = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Could not load existing FAISS index: {e}")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "TNAU Agricultural Schemes RAG API is running"}


@app.post("/upload-pdf/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global vector_store

    all_chunks = []

    for uploaded_file in files:
        if not uploaded_file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{uploaded_file.filename} is not a PDF file."
            )

        temp_path = UPLOAD_DIR / Path(uploaded_file.filename).name

        try:
            content = await uploaded_file.read()
            temp_path.write_bytes(content)

            loader = PyPDFLoader(str(temp_path))
            documents = loader.load()

            chunks = text_splitter.split_documents(documents)

            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.filename

            all_chunks.extend(chunks)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No readable PDF content found.")

    try:
        if vector_store is None:
            vector_store = FAISS.from_documents(all_chunks, embeddings)
        else:
            vector_store.add_documents(all_chunks)

        vector_store.save_local(str(FAISS_DIR))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "PDFs processed and saved successfully",
        "files_uploaded": len(files),
        "chunks_created": len(all_chunks)
    }


@app.post("/query/")
async def query_documents(request: QueryRequest):
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Upload PDF documents before asking questions."
        )

    try:
        results = vector_store.similarity_search(request.question, k=4)

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in results
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an agricultural government schemes assistant
for Tamil Nadu farmers.

Answer questions using only the supplied document context.
If the answer is not present in the documents, clearly say:
'I could not find this information in the uploaded documents.'

Explain schemes in simple language.
Do not invent eligibility criteria, subsidy amounts, deadlines,
or government benefits.

Document context:
{context}"""
            ),
            ("human", "{question}")
        ])

        chain = prompt | llm
        response = await chain.ainvoke({
            "context": context,
            "question": request.question
        })

        sources = [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content
            }
            for doc in results
        ]

        return {
            "question": request.question,
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

