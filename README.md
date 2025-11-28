# Chatbot PDF - Django RAG Application

Upload PDF, ask questions, get AI answers powered by Gemini + FAISS vector search.

## **Quick Start**

### 1. Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Environment
Copy .ven  .env, update with API keys:
```env
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_id_here
```

### 3. Run
```powershell
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Visit http://127.0.0.1:8000/

## **Features**

- **Chat** (/): Ask questions about PDF documents
- **Admin** (/admin/): Manage documents, users, Q&A history
- **Upload** (/upload/): Admin only - upload PDF files
- **Auth** (/login/, /register/): User authentication

## **Tech Stack**

- Django 5.0.6
- SentenceTransformers (embeddings)
- FAISS (vector search)
- Google Gemini API
- SQLite/MySQL

## **Pages**

| URL | Purpose |
|-----|---------|
| / | Chat interface |
| /login/ | User login |
| /register/ | User registration |
| /upload/ | Upload PDF (admin only) |
| /account/ | Manage users (admin only) |
| /admin/ | Django admin panel |

## **Troubleshooting**

| Issue | Solution |
|-------|----------|
| FAISS fails on Windows | `conda install -c conda-forge faiss` |
| **API key error** | Check `.env` has correct Google API keys. Web search will gracefully fallback if keys invalid. |
| **Gemini model not found (404)** | Using `gemini-1.5-flash` now (newer model). If issue persists, check API key quota. |
| Database error | SQLite by default. For MySQL, set `DB_*` env vars |
| PDF text empty | PDF may be encrypted or scanned image |
| Port 8000 in use | `python manage.py runserver 8080` |
| No documents processed | Upload PDF in `/upload/`, system auto-processes. Check logs. |

## **Key Files**

- pythonweb/settings.py - Django config (loads .env)
- home/views.py - Views for chat, upload, auth
- home/rag.py - PDF extraction, embeddings, Gemini API
- home/models.py - Document, Answer, ProcessedDocument
- .env - Environment variables (**don't commit**)

## **Environment Variables**

```env
GEMINI_API_KEY=your_gemini_key
GOOGLE_API_KEY=your_google_key
GOOGLE_CSE_ID=your_cse_id
DB_ENGINE=django.db.backends.sqlite3  # or .mysql
```

## **Database Setup**

### SQLite (Default - Development)
No configuration needed. Uses `db.sqlite3` file by default.

### MySQL (Production)

1. **Create database:**
```sql
CREATE DATABASE chat_bot;
```

2. **Update `.env`:**
```env
DB_ENGINE=django.db.backends.mysql
DB_NAME=chat_bot
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
```

3. **Install MySQL driver:**
```powershell
pip install mysqlclient
```

4. **Run migrations:**
```powershell
python manage.py migrate
```
