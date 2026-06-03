import os
import fitz
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

st.set_page_config(page_title="AI PDF Chat", layout="wide")
st.title("AI PDF Chat")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.chat_input("Ask a question about the PDF")

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        text += page.get_text()

    return text

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return result.embeddings[0].values


def build_vector_db(chunks):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("pdf_chunks")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )

    return collection

def retrieve_context(collection, question, n_results=4):
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    return "\n\n".join(results["documents"][0])

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
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)

    with st.spinner("Creating vector database..."):
        collection = build_vector_db(chunks)

    st.success("PDF processed. You can now ask questions.")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            context = retrieve_context(collection, question)
            answer = answer_question(context, question)

        with st.chat_message("assistant"):
            st.write(answer)