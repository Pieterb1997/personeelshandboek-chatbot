# 💬 Personeelshandboek Chatbot

Een simpele Streamlit-webapp waarmee medewerkers vragen kunnen stellen over het personeelshandboek.

## 🔧 Functionaliteit

- PDF-handboek wordt ingeladen en doorzocht via AI
- Vraag-antwoord interface via chat
- Gebouwd met:
  - Streamlit
  - Langchain
  - OpenAI GPT-4
  - FAISS

## 🚀 Deployment (Render.com)

1. Zet deze repo op GitHub
2. Maak een nieuw Web Service aan op [https://render.com](https://render.com)
3. Gebruik deze instellingen:

**Build Command**
```
pip install -r requirements.txt
```

**Start Command**
```
streamlit run app.py --server.port 10000
```

**Environment Variable**
```
OPENAI_API_KEY = je-api-key
```

## 📁 Bestanden

- `app.py`: de hoofdapplicatie
- `requirements.txt`: afhankelijkheden
- `.gitignore`: voorkomt dat geheime bestanden geüpload worden

## 📝 Opmerking

Zorg dat je zelf `personeelshandboek.pdf` toevoegt vóór je de app start.
