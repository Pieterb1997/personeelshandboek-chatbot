# 💬 Personeelshandboek Chatbot (met uploadfunctie)

Een Streamlit-app waarmee medewerkers kunnen chatten met een AI die het personeelshandboek kent.

## 🔧 Features

- Chatinterface met GPT-4
- Handboek wordt ingeladen vanuit PDF
- Uploadmogelijkheid om een nieuw handboek toe te voegen (via de browser)
- Werkt met LangChain + OpenAI

## 🚀 Deployment op Render.com

1. Zet deze repo op GitHub
2. Maak een Web Service aan op [https://render.com](https://render.com)
3. Gebruik:

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
OPENAI_API_KEY = je-openai-api-key
```

Na deployment kun je het handboek uploaden via de zijbalk.

## 📁 Bestanden

- `app.py`
- `requirements.txt`
- `.gitignore`
