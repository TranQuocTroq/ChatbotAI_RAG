import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import src.core.win_fix
from src.core.rag_pipeline import RAGPipeline


def run_test():
    print("=" * 50)
    print("Testing DocBrain RAG Pipeline with Local LLM")
    print("=" * 50)

    pipeline = RAGPipeline()
    session_id = "session_582b8d994da8"
    session_docs = ["20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf"]

    test_questions = [
        "Who are you?",
        "What is your name?",
        "What can you do for me?",
        "Haha, interesting",
        "Do you remember my name?",
        "Hello",
        "What is this document about?",
        "What is Vinamilk's revenue for 2025?",
        "I am feeling sad",
        "Encourage me",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        res = pipeline.ask(query=q, session_id=session_id, session_docs=session_docs, top_k=3)
        print(f"[Latency] {res['execution_time_sec']}s | [Confidence] {res.get('confidence', 0) * 100}% | API Used: {res.get('used_api', False)}")
        print(f"[Answer]  {res['answer']}")
        for src in res.get("sources", []):
            print(f"   [{src['id']}] {src['source_file']} (Page {src['page']}) - Score: {src['relevance_score']}")
        print("-" * 60)


if __name__ == "__main__":
    run_test()
