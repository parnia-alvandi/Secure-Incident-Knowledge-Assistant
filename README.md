# Secure Incident Knowledge Assistant

A local RAG-based assistant for answering questions about cybersecurity incident documents (PDF, DOCX, TXT).

The system uses local embeddings, ChromaDB, and Ollama to retrieve relevant information and generate answers based on document evidence. Each answer includes its source and relevant excerpt. If sufficient evidence is unavailable, the system reports it instead of guessing.

## Requirements

- Windows 11
- Python 3.10+
- Ollama
- 16 GB RAM recommended
- CPU only (no GPU required)

## Installation

```powershell
git clone <repository-url>
cd <repository-folder>

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Install CPU-only PyTorch if required:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Download the local LLM:

```powershell
ollama pull qwen2.5:3b-instruct
```

## Documents and Indexing

Copy the project documents into:

```text
data/docs/
```

Update `data/manifest.yaml` if necessary.

For the initial embedding model download, set the following option in `config.yaml`:

```yaml
embedding:
  offline: false
```

Build and verify the index:

```powershell
python scripts/build_index.py
python scripts/verify_index.py
```

After downloading the embedding model, set `embedding.offline` back to `true`.

## Running

Ask a question:

```powershell
python scripts/ask.py --question "Your question here"
```

Interactive mode:

```powershell
python scripts/ask.py
```

Retrieval only (without Ollama):

```powershell
python scripts/ask.py --question "Your question here" --no-llm
```

JSON output:

```powershell
python scripts/ask.py --question "Your question here" --json
```

## Testing

```powershell
pytest -q
```

Evaluation results for five answerable questions and one unanswerable question are stored in `docs/demo/`.

## Response Statuses

- **ANSWERED:** Answer supported by retrieved evidence and valid citations.
- **INSUFFICIENT EVIDENCE:** Available documents do not contain sufficient evidence.
- **UNVERIFIED:** The answer does not contain a valid citation.

## Notes

- Document processing, embeddings, retrieval, and LLM inference run locally.
- Initial model downloads require an internet connection.
- English documents and questions are currently supported.
- Scanned PDFs without a text layer are not supported.
- Hybrid search and Cross-Encoder re-ranking are under evaluation and are not part of the current baseline pipeline.

---

# دستیار دانش اسناد رخدادهای امنیتی

یک سیستم محلی مبتنی بر معماری RAG برای پاسخ‌گویی به سؤالات مرتبط با اسناد رخدادهای امنیت سایبری (PDF، DOCX و TXT).

این سیستم با استفاده از Embedding محلی، پایگاه داده ChromaDB و مدل زبانی Ollama، اطلاعات مرتبط را بازیابی کرده و بر اساس شواهد موجود در اسناد پاسخ تولید می‌کند. هر پاسخ شامل منبع و بخش مرتبط از متن سند است. در صورت نبود شواهد کافی، سیستم از حدس زدن خودداری می‌کند.

## پیش‌نیازها

- ویندوز ۱۱
- Python نسخه 3.10 یا بالاتر
- Ollama
- حداقل ۱۶ گیگابایت RAM توصیه می‌شود.
- پردازنده CPU (بدون نیاز به GPU)

## نصب پروژه

```powershell
git clone <repository-url>
cd <repository-folder>

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

در صورت نیاز، نسخه CPU کتابخانه PyTorch را نصب کنید:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

دریافت مدل زبانی:

```powershell
ollama pull qwen2.5:3b-instruct
```

## آماده‌سازی اسناد و ساخت ایندکس

اسناد موردنظر را در مسیر زیر قرار دهید:

```text
data/docs/
```

در صورت نیاز، فایل `data/manifest.yaml` را به‌روزرسانی کنید.

برای دانلود اولیه مدل Embedding، مقدار زیر را در `config.yaml` تنظیم کنید:

```yaml
embedding:
  offline: false
```

سپس دستورات زیر را اجرا کنید:

```powershell
python scripts/build_index.py
python scripts/verify_index.py
```

پس از دانلود مدل، مقدار `embedding.offline` را مجدداً روی `true` قرار دهید.

## اجرای پروژه

پرسیدن سؤال:

```powershell
python scripts/ask.py --question "سؤال موردنظر"
```

حالت تعاملی:

```powershell
python scripts/ask.py
```

بازیابی اطلاعات بدون استفاده از Ollama:

```powershell
python scripts/ask.py --question "سؤال موردنظر" --no-llm
```

دریافت خروجی JSON:

```powershell
python scripts/ask.py --question "سؤال موردنظر" --json
```

## اجرای تست‌ها

```powershell
pytest -q
```

نتایج ارزیابی پنج سؤال دارای پاسخ و یک سؤال بدون پاسخ در مسیر `docs/demo/` ذخیره می‌شوند.

## وضعیت پاسخ‌ها

- **ANSWERED:** پاسخ دارای شواهد بازیابی‌شده و ارجاع معتبر است.
- **INSUFFICIENT EVIDENCE:** اسناد موجود شواهد کافی برای پاسخ‌گویی ندارند.
- **UNVERIFIED:** پاسخ فاقد ارجاع معتبر است.

## توضیحات

- پردازش اسناد، تولید Embedding، بازیابی اطلاعات و اجرای مدل زبانی به‌صورت محلی انجام می‌شود.
- دانلود اولیه مدل‌ها به اینترنت نیاز دارد.
- در حال حاضر اسناد و سؤالات انگلیسی پشتیبانی می‌شوند.
- فایل‌های PDF اسکن‌شده که متن قابل استخراج ندارند، پشتیبانی نمی‌شوند.
- قابلیت Hybrid Search و Cross-Encoder در حال ارزیابی هستند و هنوز در مسیر اصلی Baseline RAG استفاده نمی‌شوند.