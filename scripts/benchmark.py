"""Hardware benchmark: embedding, reranker, and Ollama LLM speed + RAM.

Run from the project root:   python scripts/benchmark.py
Optional:                    python scripts/benchmark.py --llm-model qwen2.5:7b-instruct
                             python scripts/benchmark.py --skip-llm
                             python scripts/benchmark.py --skip-models
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import psutil

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sia.config import load_settings  # noqa: E402
from sia.logging_setup import setup_logging  # noqa: E402

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 1024 / 1024


def bench_embedding(model_name: str) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"\n=== Embedding: {model_name} ===")
    before = rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"Load time: {load_s:.1f}s | RAM added: ~{rss_mb() - before:.0f} MB")

    texts = [
        "Ransomware encrypts files and demands payment. Containment means isolating "
        "affected hosts from the network to stop lateral movement. " * 4
    ] * 64
    model.encode(texts[:4])  # warm-up
    t0 = time.perf_counter()
    model.encode(texts, batch_size=16)
    dt = time.perf_counter() - t0
    print(f"Encode: {len(texts) / dt:.1f} chunks/s ({len(texts)} chunks in {dt:.2f}s)")


def bench_reranker(model_name: str) -> None:
    from sentence_transformers import CrossEncoder

    print(f"\n=== Reranker: {model_name} ===")
    before = rss_mb()
    t0 = time.perf_counter()
    model = CrossEncoder(model_name, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"Load time: {load_s:.1f}s | RAM added: ~{rss_mb() - before:.0f} MB")

    query = "What are the phases of incident response?"
    doc = (
        "Incident response includes preparation, detection and analysis, containment, "
        "eradication, recovery, and post-incident activity. " * 4
    )
    pairs = [(query, doc)] * 20
    model.predict(pairs[:2])  # warm-up
    t0 = time.perf_counter()
    model.predict(pairs)
    dt = time.perf_counter() - t0
    print(f"Rerank: {len(pairs) / dt:.1f} pairs/s (20 pairs in {dt:.2f}s)")


def _report(resp: dict) -> None:
    """Print timing numbers from an Ollama response (durations are in nanoseconds)."""
    load_s = resp.get("load_duration", 0) / 1e9
    p_cnt, p_dur = resp.get("prompt_eval_count", 0), resp.get("prompt_eval_duration", 0) / 1e9
    g_cnt, g_dur = resp.get("eval_count", 0), resp.get("eval_duration", 0) / 1e9
    print(f"Model load: {load_s:.1f}s")
    if p_dur > 0:
        print(f"Prefill:    {p_cnt} tokens in {p_dur:.2f}s ({p_cnt / p_dur:.1f} tok/s)")
    if g_dur > 0:
        print(f"Generation: {g_cnt} tokens in {g_dur:.2f}s ({g_cnt / g_dur:.1f} tok/s)")


def bench_llm(model_name: str, num_ctx: int, num_predict: int) -> None:
    import ollama

    print(f"\n=== LLM (Ollama): {model_name} | short prompt ===")
    prompt = (
        "Explain in a short paragraph what 'containment' means during a computer "
        "security incident, and give two examples."
    )
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
        )
    except Exception as exc:  # noqa: BLE001 - report any connection/model error
        print(f"LLM benchmark FAILED: {exc}")
        print("Is Ollama running? Is the model pulled? (ollama list)")
        return
    _report(resp)
    print("Sample output:", resp["message"]["content"][:200].replace("\n", " "), "...")


def bench_llm_long(model_name: str, num_ctx: int) -> None:
    """Long-prompt test (~1500 tokens) that mimics a real RAG prompt with 5 chunks.

    A random nonce is placed at the START of the prompt so Ollama cannot reuse a
    cached prefix; otherwise prefill would be measured as artificially fast.
    """
    import ollama

    print(f"\n=== LLM (Ollama): {model_name} | long prompt (RAG-like) ===")
    paragraph = (
        "Incident response is the organized approach to addressing and managing the "
        "aftermath of a security breach. Containment limits the spread of an attack by "
        "isolating affected systems, disabling compromised accounts, and blocking "
        "malicious network traffic. Eradication removes the root cause, such as malware "
        "or unauthorized access, and recovery restores systems to normal operation. "
    )
    context = "\n\n".join(f"[Doc {i + 1}, p.{i + 1}]\n{paragraph * 3}" for i in range(5))
    prompt = (
        f"Session {uuid.uuid4()}\n"
        "Answer ONLY from the context below.\n\n"
        f"Context:\n{context}\n\n"
        "Question: What does containment mean?"
    )
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": max(num_ctx, 4096), "num_predict": 60},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Long-prompt benchmark FAILED: {exc}")
        return
    _report(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="SIA hardware benchmark")
    parser.add_argument("--llm-model", default=None, help="override LLM model name")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-models", action="store_true", help="skip embedding/reranker")
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings.logging.level)

    print(f"CPU logical cores: {psutil.cpu_count()} | "
          f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB | "
          f"Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    if not args.skip_models:
        bench_embedding(settings.models.embedding)
        bench_reranker(settings.models.reranker)
    if not args.skip_llm:
        model = args.llm_model or settings.llm.model
        bench_llm(model, settings.llm.num_ctx, settings.llm.num_predict)
        bench_llm_long(model, settings.llm.num_ctx)


if __name__ == "__main__":
    main()