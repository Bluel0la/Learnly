# Learnly API 📚🧠

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112.2-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-SQLAlchemy_2.0-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-Migrations-red?style=for-the-badge&logo=sqlite&logoColor=white)

**Learnly API** is an AI-powered educational and study companion backend built with **FastAPI**, **PostgreSQL**, and **OpenAI GPT-4o**. It empowers learners by automatically converting uploaded study documents (PDFs, DOCX, PPTX, Images) and topic prompts into interactive flashcard decks, adaptive spaced-repetition drills, customized quizzes, simulated examinations, and interactive AI study chats.

---

## 🌟 Key Features

### 🔐 Authentication & Account Management
* **JWT Authentication**: Secure user signup, login, session token validation, and refresh token handling.
* **Token Blacklisting**: Logout and security mechanisms using token revocation lists.
* **Profile Management**: Profile updates, password changes, and account deletion endpoints.

### 🤖 AI Study Assistant & Chat (`/api/v1/chat`)
* **Interactive AI Tutor**: Session-based chat assistant powered by OpenAI GPT-4o for answering questions and explaining difficult concepts.
* **Multi-Format Document Parsing**: Text extraction from PDFs (`PyPDF2`), Word documents (`python-docx`), PowerPoint slides (`python-pptx`), and images (`Pillow`).

### 🃏 Smart Flashcards & Adaptive Learning (`/api/v1/flashcards`)
* **AI Deck Generation**: Automatic creation of structured flashcards directly from uploaded study notes or topic summaries.
* **Spaced Repetition & Progress Tracking**: Categorizes cards as *Unstudied*, *Hard*, or *Bookmarked* to optimize review sessions.
* **Adaptive Drills**: Regenerates practice decks based on user performance.
* **Deck Quizzes**: Interactive quiz modes with automated evaluation for specific decks.

### 📝 Interactive Quizzes & Examinations (`/api/v1/quiz`)
* **Dynamic Quiz Sessions**: Batch question retrieval, real-time response submission, and automated score calculation.
* **Simulated Exams**: Comprehensive exam creation matching target difficulty levels.
* **Performance Analytics**: Analytics breakdown by topic (`/quiz/topics`) and performance summary tracking (`/quiz/performance`).
* **Session Review**: Detailed post-quiz reviews to inspect correct answers and explanations.

### ⚡ Infrastructure & Reliability
* **Rate Limiting**: Built-in request rate limiting to protect API endpoints and manage LLM cost overhead.
* **Database Migrations**: Managed database schema evolution using Alembic.
* **Custom Error Handling**: Structured exception middleware returning clear JSON error responses.

---

## 🛠️ Tech Stack

* **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
* **ASGI Server**: [Uvicorn](https://www.uvicorn.org/)
* **Database**: PostgreSQL
* **ORM & Migrations**: [SQLAlchemy 2.0](https://www.sqlalchemy.org/) & [Alembic](https://alembic.sqlalchemy.org/)
* **AI / LLM Integration**: [OpenAI Python SDK](https://github.com/openai/openai-python) (GPT-4o / GPT-4o-mini)
* **Authentication**: PyJWT / `python-jose`, `passlib` with `bcrypt`
* **Document Parsing**: `PyPDF2`, `python-docx`, `python-pptx`, `Pillow`, `pyspellchecker`
* **Configuration**: `pydantic-settings` & `python-dotenv`

---

## 📂 Project Structure

```text
learnly_api/
├── alembic/                # Database migration scripts & configuration
├── api/
│   ├── core/               # App configuration, security, rate limiters, exception handlers
│   ├── db/                 # Database session setup and base model configuration
│   ├── utils/              # File parsers (PDF, DOCX, PPTX), prompt helpers, text extractors
│   └── v1/
│       ├── models/         # SQLAlchemy ORM models (User, Deck, Card, Quiz, Chat, Tokens)
│       ├── routes/         # FastAPI endpoint routers (auth, chat, flashcards, quiz)
│       ├── schemas/        # Pydantic schemas for request validation & response serializing
│       └── services/       # Core business logic & OpenAI service integrations
├── main.py                 # Application entrypoint & CORS middleware setup
├── requirements.txt        # Python package dependencies
├── alembic.ini             # Alembic migration configuration
└── .env.example            # Environment variables template
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your machine:
* **Python 3.10+**
* **PostgreSQL** database server
* **OpenAI API Key**

### 1. Clone the Repository

```bash
git clone https://github.com/Bluel0la/Learnly.git
cd learnly_api
```

### 2. Set Up Virtual Environment

**Windows (PowerShell / CMD):**
```powershell
py -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Fill in your configuration settings in `.env`:

```ini
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_TOKENS_SUMMARY=512
OPENAI_MAX_TOKENS_CHAT=1024

# Database
DB_URL=postgresql+psycopg2://user:password@localhost:5432/learnly

# JWT Authentication
SECRET_KEY=your_super_secret_jwt_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Rate Limiting
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW_SECONDS=3600

# Quiz Difficulty Thresholds
QUIZ_PRO_THRESHOLD=85.0
QUIZ_MEDIUM_THRESHOLD=60.0
```

### 5. Database Setup & Migrations

Ensure your PostgreSQL service is running and the database specified in `DB_URL` exists. Run the Alembic migrations to create all required database tables:

```bash
alembic upgrade head
```

---

## 🏃 Running the Application

Start the development server using **Uvicorn**:

```bash
uvicorn main:app --port 7001 --reload
```

Or run `main.py` directly:

```bash
python main.py
```

The API will be live at `http://localhost:7001`.

---

## 📖 API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, explore and test endpoints directly in your browser:

* **Swagger UI**: [http://localhost:7001/docs](http://localhost:7001/docs)
* **ReDoc**: [http://localhost:7001/redoc](http://localhost:7001/redoc)

---

## 📍 API Endpoints Summary

Base URL Prefix: `/api/v1`

| Module | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **Auth** | `POST` | `/auth/signup` | Register a new user account |
| **Auth** | `POST` | `/auth/login` | Authenticate user and retrieve JWT token |
| **Auth** | `GET` | `/auth/me` | Fetch current user details |
| **Auth** | `POST` | `/auth/logout` | Revoke session tokens |
| **Auth** | `PUT` | `/auth/update` | Update profile information |
| **Auth** | `POST` | `/auth/change-password` | Change account password |
| **Chat** | `POST` | `/chat/start-session` | Initialize a new AI chat session |
| **Chat** | `POST` | `/chat/send-message` | Send prompt to AI tutor and receive response |
| **Chat** | `POST` | `/chat/extract-text/` | Upload document (PDF/DOCX/PPTX/Image) for text extraction |
| **Chat** | `GET` | `/chat/sessions` | List all active/past user chat sessions |
| **Flashcards** | `POST` | `/flashcards/decks/` | Create a new flashcard deck |
| **Flashcards** | `POST` | `/flashcards/decks/{deck_id}/generate-flashcards/` | Generate flashcards using AI from study materials |
| **Flashcards** | `GET` | `/flashcards/decks/{deck_id}/practice` | Retrieve practice cards for study session |
| **Flashcards** | `POST` | `/flashcards/cards/{card_id}/submit-response` | Submit response to update card difficulty metrics |
| **Quiz** | `POST` | `/quiz/start` | Initialize a new quiz session |
| **Quiz** | `GET` | `/quiz/questions/{session_id}` | Retrieve batch of quiz questions |
| **Quiz** | `POST` | `/quiz/{session_id}/submit` | Submit answers for scoring |
| **Quiz** | `GET` | `/quiz/performance` | Retrieve overall performance analytics |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
