import os
import fitz
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai
import hashlib

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

st.set_page_config(page_title="AI PDF Chat", layout="wide")
st.title("AI PDF Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("**Sources:**")
            st.write(", ".join([f"Page {p}" for p in message["sources"]]))

question = st.chat_input("Ask a question about the PDF")

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def extract_text_from_pdf(pdf_bytes):
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages

def chunk_text(pages, chunk_size=800, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return result.embeddings[0].values


def build_vector_db(chunks, file_hash):
    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name=f"pdf_{file_hash}"
    )

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{"page": chunk["page"]}]
        )

    return collection

def retrieve_context(collection, question, n_results=4):
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    source_pages = []

    for doc, metadata in zip(documents, metadatas):
        page = metadata.get("page", "Unknown")
        context_parts.append(f"[Page {page}]\n{doc}")
        source_pages.append(page)

    context = "\n\n".join(context_parts)
    unique_pages = sorted(set(source_pages))

    return context, unique_pages

def answer_question(context, question):
    prompt = f"""
You are a helpful assistant. Answer the user's question using only the PDF context below.
If the answer is not in the context, say you cannot find it in the PDF.

PDF context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(pdf_bytes)

    if st.session_state.get("file_hash") != file_hash:
        st.session_state.messages = []

        with st.spinner("Reading PDF and creating embeddings..."):
            pages = extract_text_from_pdf(pdf_bytes)
            chunks = chunk_text(pages)
            collection = build_vector_db(chunks, file_hash)

        st.session_state.file_hash = file_hash
        st.session_state.collection = collection
    else:
        collection = st.session_state.collection
        st.info("Using cached PDF embeddings.")
    
if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):
        context, source_pages = retrieve_context(collection, question)
        answer = answer_question(context, question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_pages
    })

    with st.chat_message("assistant"):
        st.write(answer)

        if source_pages:
            st.markdown("**Sources:**")
            st.write(", ".join([f"Page {p}" for p in source_pages]))