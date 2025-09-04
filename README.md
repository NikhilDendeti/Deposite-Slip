# Deposit Slip Processing System

Backend: FastAPI + SQLAlchemy + OCR (Tesseract/OpenCV, optional PaddleOCR) + optional OpenAI LLM. Frontend: Streamlit.

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
│   ├── llm_parser.py
│   ├── slip_rule_parser.py
│   └── validation.py
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── conftest.py
│   └── test_app.py
├── uploads/                  # input images/PDFs are saved here
├── outputs/                  # batch results are written here
├── batch_eval.py             # batch evaluation over uploads
├── requirements.txt
├── test.db                   # default SQLite db (dev)
└── README.md
```

## Prerequisites

- Python 3.10+
- Tesseract OCR (Linux example: `sudo apt install tesseract-ocr tesseract-ocr-eng`)
- For PDF rendering, PyMuPDF is used (installed via requirements). No system install needed.

Optional:
- Google Cloud Vision API key if using GCV mode
- OpenAI account/key if using LLM/Vision modes
- PaddlePaddle/PaddleOCR (CPU) are included in `requirements.txt` for some environments

## Setup

```bash
cd "$(dirname "$0")"
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment variables (.env optional)

Create a `.env` file in the project root if needed:

```
DATABASE_URL=sqlite:///./test.db

# Optional: enable Google Cloud Vision OCR (REST, API key)
GOOGLE_CLOUD_VISION_API_KEY=your_gcv_api_key

# Optional: enable OpenAI-powered parsing/vision
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
```

Notes:
- If `OPENAI_API_KEY` is not set, the app still works with pure OCR mode.
- `DATABASE_URL` defaults to `sqlite:///./test.db` when not provided.

## Run

Backend (FastAPI):

```bash
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend (Streamlit):

```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open Streamlit at http://localhost:8501 and set API base URL to `http://localhost:8000`.

## Batch evaluation (optional)

Run OCR/LLM/Vision modes over images in `uploads/` and write results to `outputs/`:

```bash
source venv/bin/activate
python batch_eval.py \
  --uploads-dir uploads \
  --output outputs/results.jsonl \
  --modes ocr llm vision \
  --concurrency 2
```

Outputs:
- `outputs/results.jsonl`: one JSON record per image with per-mode results
- `outputs/summary.json`: compact table of key fields per mode

## API quick reference

- POST `/deposit-slips/upload` (multipart form)
  - fields: `collection_id` (int), optional `manual_amount` (float), `manual_date` (YYYY-MM-DD), `mode` in `ocr|llm|vision|gcv`
  - file: `file` (image or PDF)
- GET `/deposit-slips` (optional `status` filter)
- POST `/collections`
- GET `/collections`

Images (PNG/JPG/TIFF/BMP) and PDFs are supported. On upload, files are saved under `uploads/`.

### Google Cloud Vision mode

- Select `gcv` in the Streamlit "Processing mode" dropdown or pass `mode=gcv` to the upload API.
- The client uses the REST `images:annotate` endpoint with `DOCUMENT_TEXT_DETECTION`.
- Reference: Google Cloud Vision REST API docs [`https://cloud.google.com/vision/docs/reference/rest`](https://cloud.google.com/vision/docs/reference/rest)

## Testing

```bash
source venv/bin/activate
pytest -q
```
