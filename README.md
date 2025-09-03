# Deposit Slip Processing System

Backend: FastAPI + SQLAlchemy + OCR (Tesseract). Frontend: Streamlit.

## Project structure

```
deposit-slip-system/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   ├── database.py
│   ├── ocr_service.py
│   ├── validation.py
│   └── auth.py
├── frontend/
│   └── streamlit_app.py
├── requirements.txt
├── .env (optional)
└── README.md
```

## Prerequisites

- Python 3.10+
- Tesseract OCR installed on your system (e.g., `sudo apt install tesseract-ocr`)

## Setup

```bash
cd "$(dirname "$0")"
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optionally create `.env`:

```
DATABASE_URL=sqlite:///./test.db
SECRET_KEY=replace_with_strong_secret
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

## Run

Backend:

```bash
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open Streamlit at http://localhost:8501 and set API base URL to `http://localhost:8000`.


