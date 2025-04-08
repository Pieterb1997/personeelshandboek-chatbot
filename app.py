import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Pagina instellingen ---
st.set_page_config(page_title="HR Chatbot", page_icon="💬")
st.title("💬 Chat met het Personeelshandboek")

# --- OpenAI API key ophalen ---
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")  # Voor Render
    if not openai_api_key:
        openai_api_key = st.secrets["OPENAI_API_KEY"]  # Voor lokaal
except Exception:
    openai_api_key = None

if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt.")
    st.stop()

# --- Check of PDF aanwezig is ---
pdf_pad = "personeelshandboek.pdf"

if not os.path.exists(pdf_pad):
    st.error("📄 Het bestand 'personeelshandboek.pdf' is niet gevonden in de projectmap.")
    st.stop()

# --- Verwerk PDF tot kennisbank ---
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

qa = load_pdf_and_build_qa(pdf_pad)

# --- Chat interface ---
vraag = st.chat_input("Stel je vraag over het handboek...")

if vraag:
    st.chat_message("user").markdown(vraag)
    with st.chat_message("assistant"):
        response = qa.invoke(vraag)
        st.markdown(response["result"])
