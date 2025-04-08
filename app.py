import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os

# --- Streamlit UI setup ---
st.set_page_config(page_title="HR Chatbot", page_icon="💬")
st.title("💬 Chat met het Personeelshandboek")

# --- OpenAI key ophalen ---
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ OpenAI API key ontbreekt.")
    st.stop()

# --- Laad PDF en bouw kennisbank ---
@st.cache_resource
def load_pdf_and_build_qa(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Laad kennisbank (PDF moet bestaan!)
if not os.path.exists("personeelshandboek.pdf"):
    st.warning("📄 Voeg je 'personeelshandboek.pdf' toe aan de projectmap.")
    st.stop()

qa = load_pdf_and_build_qa("personeelshandboek.pdf")

# --- Chat input ---
vraag = st.chat_input("Stel je vraag over het handboek...")

if vraag:
    st.chat_message("user").markdown(vraag)
    with st.chat_message("assistant"):
        response = qa.run(vraag)
        st.markdown(response)
