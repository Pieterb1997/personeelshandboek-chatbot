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
    openai_api_key = os.environ.get("OPENAI_API_KEY")  # Voor Render (env var)
    if not openai_api_key:
        openai_api_key = st.secrets["OPENAI_API_KEY"]  # Voor lokale dev
except Exception:
    openai_api_key = None

if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe als environment variable op Render.")
    st.stop()

# --- Upload nieuw handboek ---
st.sidebar.subheader("📄 Handboek uploaden")
uploaded_file = st.sidebar.file_uploader("Upload een nieuw handboek (PDF)", type=["pdf"])
if uploaded_file:
    with open("personeelshandboek.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Nieuw handboek is opgeslagen ✅")
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# --- Functie om kennisbank op te bouwen ---
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

# --- Controleer of PDF aanwezig is ---
if not os.path.exists("personeelshandboek.pdf"):
    st.warning("📄 Upload eerst het personeelshandboek via de zijbalk.")
    st.stop()

qa = load_pdf_and_build_qa("personeelshandboek.pdf")

# --- Chat interface ---
vraag = st.chat_input("Stel je vraag over het handboek...")

if vraag:
    st.chat_message("user").markdown(vraag)
    with st.chat_message("assistant"):
        response = qa.invoke(vraag)
        st.markdown(response["result"])
