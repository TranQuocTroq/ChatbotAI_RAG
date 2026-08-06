import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import src.core.win_fix
from src.core.rag_pipeline import RAGPipeline

def run_session_ingestion():
    print("==================================================")
    print(" Advanced Hybrid Ingestion (Dense + BM25) - Session 1")
    print("==================================================")

    pipeline = RAGPipeline()
    sample_pdf = Path("data/raw/vinamilk/vinamilk_bctn_2025.pdf")

    if not sample_pdf.exists():
        print(f"[!] Sample file {sample_pdf} does not exist. Skipped.")
        return

    res = pipeline.process_and_ingest_file_for_session("session_1", str(sample_pdf))
    print(f" Hybrid Ingestion completed for Session 1: {res.get('total_chunks', 0)} chunks saved at data/sessions_vector_stores/session_1/")

if __name__ == "__main__":
    run_session_ingestion()
