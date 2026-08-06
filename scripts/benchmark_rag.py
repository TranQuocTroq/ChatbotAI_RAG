import os
import time
import json
import logging
from pathlib import Path

# Log configuration for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure working directory is project root
if not Path("src").exists():
    os.chdir("..")

from src.core.rag_pipeline import RAGPipeline


def run_benchmark():
    # Disable logging for some sub-libraries for cleaner console
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    print("\n" + "=" * 50)
    print(" START RAG PIPELINE BENCHMARK")
    print("=" * 50 + "\n")

    # Initialize the pipeline
    start_init = time.time()
    pipeline = RAGPipeline()
    print(f"[+] RAGPipeline initialized in: {time.time() - start_init:.2f}s")

    session_id = "benchmark_session"

    # Clear cache to measure actual LLM runtime
    cache_file = Path("data/cache/semantic_cache.pkl")
    if cache_file.exists():
        cache_file.unlink()
        print("[+] Cleared old semantic cache.")

    test_file = "tai_lieu_test_tom_tat.txt"
    if Path(test_file).exists():
        print(f"[+] Loading document {test_file} into session {session_id}...")
        pipeline.process_and_ingest_file_for_session(session_id, test_file)
    else:
        print(f"[-] Not found {test_file}, running dry.")

    # 5 sample queries about the Vinamilk report, mixing Vietnamese and English
    # to also exercise cross-lingual retrieval.
    test_queries = [
        "Tổng doanh thu thuần của Vinamilk năm 2025 là bao nhiêu?",  # Vietnamese, table-related
        "Lợi nhuận sau thuế của công ty có tăng trưởng không?",       # Vietnamese, table-related
        "What is the total revenue of Vinamilk in 2025?",            # English (cross-lingual test)
        "Kế hoạch phát triển bền vững (ESG) của công ty là gì?",      # Vietnamese, text content
        "Summarize the main financial risks over the past year."     # English, long-form content
    ]

    results = []
    total_time = 0.0

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Question {i}: {query}")
        start_q = time.time()

        # Bypass HTTP, call ask() directly
        res = pipeline.ask(query, session_id=session_id)

        exec_time = res.get("execution_time_sec", time.time() - start_q)
        total_time += exec_time

        answer_preview = res.get("answer", "").replace("\n", " ")[:150] + "..."
        sources = [s["source_file"] for s in res.get("sources", [])]

        print(f"    [Latency] {exec_time:.2f}s")
        print(f"    [Sources] {len(sources)} chunk(s)")
        print(f"    [Answer]  {answer_preview}")

        results.append({
            "query": query,
            "exec_time_sec": exec_time,
            "answer_preview": answer_preview,
            "sources_count": len(sources)
        })

    print("\n" + "=" * 50)
    print("SUMMARY RESULTS")
    print("=" * 50)
    print(f"Total time for {len(test_queries)} queries: {total_time:.2f}s")
    print(f"Average time per query: {(total_time / len(test_queries)):.2f}s\n")

    # Save results to a JSON file for future comparison
    out_file = Path("benchmark_results.json")
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        old_avg = old_data.get("avg_time_sec", 0)
        new_avg = total_time / len(test_queries)
        if old_avg > 0:
            diff = ((old_avg - new_avg) / old_avg) * 100
            print(f"Compared to previous run (baseline {old_avg:.2f}s): speed improved by {diff:.1f}%")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_time_sec": total_time,
            "avg_time_sec": total_time / len(test_queries),
            "details": results
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_benchmark()
