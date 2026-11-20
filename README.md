# Secure Incident Knowledge Assistant

Local RAG + bounded ReAct assistant for computer-security incident knowledge.
(Full documentation will be completed in Phase 9.)

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
```

## Tests
```powershell
pytest
```