# FastAPI Boilerplate

This project is a boilerplate for building APIs with FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.7+.

## Setup

Follow these steps to set up the project on your local machine.

### 1. Create a Virtual Environment

Create a virtual environment to isolate the project dependencies.

```sh
python3 -m venv .venv
```
or

```sh
virtualenv venv
```

### 2. Activate Virtual Environment

Activate the virtual environment. The command differs based on your operating system:

On macOS and Linux:
```sh
source .venv/bin/activate
```

On Windows:
```sh
.venv\Scripts\activate
```

### 3. Install Project Dependencies

Install the required dependencies from requirements.txt.

```sh
pip install -r requirements.txt
```

### 4. Create a .env File 

Create a .env file by copying the provided .env.sample file. This file will hold your environment variables.

```sh
cp .env.sample .env
```

```sh
cp .env.config.sample .env
```
### 5. Run Database Migrations

Use Alembic to run database migrations. You can create a new migration and apply it or just apply existing migrations.

To create a new migration:
```sh
alembic revision --autogenerate -m 'initial migration'
```

To apply existing migrations:
```sh
alembic upgrade head
```

### 6. Start the Server

Start the FastAPI server with Uvicorn. The --reload flag will auto-reload the server on code changes.

```sh
uvicorn main:app --reload
```


## Project Structure

```graphql
.
├── alembic/                   # Alembic migrations
├── api/                       # API routes and endpoints
│   ├── v1/                    # Version 1 of the API
│   └── ...
├── core/                      # Core configurations and utilities
├── models/                    # Database models
├── schemas/                   # Pydantic schemas (data validation and serialization)
├── main.py                    # Application entry point
├── .env.sample                # Sample environment variables file
├── requirements.txt           # Project dependencies
└── ...
```
