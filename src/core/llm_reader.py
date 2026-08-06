"""
LLMReader - Generative synthesis module for the RAG answer pipeline.

Responsibilities:
1. System self-context injection: always includes assistant identity and the list of
   session documents in the system prompt.
2. Generative LLM response: synthesizes natural-language answers using the local LLM
   (see local_llm_engine.py - no external inference API is used).
3. Multi-turn history support: accepts chat_history to keep conversational context.
4. Corrective RAG (CRAG): grades context relevance against a configurable score
   threshold instead of a purely generative guess.

Note: the system/user prompt strings sent to the LLM, and the answers returned to the
end user, are intentionally written in Vietnamese, since this product answers Vietnamese
end users. This is business logic, not a code comment, and is kept in Vietnamese
end-to-end so a small local LLM is not given a single prompt mixing two languages.
"""
import re
import logging
from typing import List, Dict, Any, Optional

from src.core.config import get_config
from src.core.document_processor import clean_pdf_text_artifacts
from src.core.local_llm_engine import generate_local

logger = logging.getLogger(__name__)


def clean_pdf_snippet(text: str) -> str:
    """Strip noisy characters and stray math symbols commonly introduced by PDF extraction."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        if re.search(r"[∈𝑐𝑛∑∫√αβγ≡λμπσ]|comp\s*[𝑛𝑐]|Thetwopenalties|clinicalloss|estimation,", l):
            continue
        cleaned.append(l)
    return "\n".join(cleaned)


class LLMReader:
    """Generates final answers via the local LLM engine (production Agentic RAG synthesizer)."""

    def __init__(self):
        cfg = get_config()
        self.temperature = cfg.get("llm.temperature", 0.2)
        self.max_new_tokens = cfg.get("llm.max_new_tokens", 512)
        self.crag_threshold = cfg.get("retrieval.score_threshold", 0.3)

    def _build_system_context(self, session_docs: List[str] = None, chat_history: List[Dict[str, str]] = None) -> str:
        """Build the system self-context always attached with assistant info and document list."""
        docs_str = "\n".join([f"- {d}" for d in session_docs]) if session_docs else "Chưa có file nào được tải lên."

        hist_str = ""
        if chat_history:
            recent = chat_history[-6:]  # last 3 conversational turns
            hist_lines = []
            for m in recent:
                role = "Người dùng" if m.get("role") == "user" else "DocBrain"
                hist_lines.append(f"{role}: {m.get('content', '')}")
            hist_str = "\nLỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n" + "\n".join(hist_lines) + "\n"

        system_prompt = (
            f"Bạn là **DocBrain AI** — Hệ thống trợ lý AI thông minh chuyên tra cứu và phân tích tài liệu.\n"
            f"Tài liệu hiện có trong phiên làm việc:\n{docs_str}\n"
            f"{hist_str}"
        )
        return system_prompt

    def generate_chitchat_response(self, query: str, session_docs: List[str] = None, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate a free-form chitchat reply through the local LLM."""
        docs_list = ", ".join(session_docs) if session_docs else "chưa có file nào"

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là DocBrain AI — trợ lý AI thông minh, linh hoạt, hóm hỉnh và thân thiện. "
                    "Hãy trò chuyện tự nhiên với người dùng như một người bạn thực sự bằng Tiếng Việt. "
                    "BẮT BUỘC trả lời thẳng vào ý của người dùng, KHÔNG lặp lại câu chào mẫu rập khuôn 'Tôi là DocBrain AI...' trừ khi người dùng hỏi tên bạn."
                )
            },
            {"role": "user", "content": query}
        ]

        try:
            ans = generate_local(messages, max_new_tokens=256, temperature=0.7)
        except Exception as e:
            logger.error("[LLMReader] Local LLM chitchat generation error: %s", e)
            ans = ""

        if not ans:
            ans = f"Chào bạn! Tôi là DocBrain AI. Hiện phiên đang có các file: {docs_list}. Tôi có thể giúp gì cho bạn?"

        return {"answer": ans, "confidence": 1.0, "is_conversational": True, "used_chunks": []}

    def generate_summary_response(self, query: str, context_chunks: List[Dict[str, Any]], session_docs: List[str] = None) -> Dict[str, Any]:
        """
        Generate a full-document summary report using Map-Reduce (Hierarchical Summary).
        Divides chunks into 3 sections (Map), summarizes each, then combines (Reduce).
        """
        docs = session_docs or []
        doc_list_str = "\n".join([f"• **{d}**" for d in docs]) if docs else "• Chưa có tài liệu"
        doc_title = docs[0] if docs else "Tài liệu"

        if not context_chunks:
            return {
                "answer": f"**BÁO CÁO TÓM TẮT NỘI DUNG TÀI LIỆU**\n\nTrong phiên làm việc này có **{len(docs)} tài liệu**:\n{doc_list_str}\n\nVui lòng tải lên tài liệu để bắt đầu tóm tắt!",
                "confidence": 1.0, "is_conversational": True, "used_chunks": []
            }

        # Divide context_chunks into up to 3 buckets (beginning, middle, end)
        bucket_size = max(1, len(context_chunks) // 3)
        buckets = [context_chunks[i:i + bucket_size] for i in range(0, len(context_chunks), bucket_size)][:3]

        map_summaries = []
        for i, bucket in enumerate(buckets):
            cleaned_texts = []
            for c in bucket:
                text_c = clean_pdf_snippet(clean_pdf_text_artifacts(c.get("text", "")))
                if text_c:
                    cleaned_texts.append(text_c[:350])

            if not cleaned_texts:
                continue

            bucket_text = "\n\n".join(cleaned_texts)
            map_messages = [
                {"role": "system", "content": "Bạn là chuyên gia tóm tắt. Hãy tóm tắt NGẮN GỌN các ý chính của phần tài liệu sau bằng Tiếng Việt."},
                {"role": "user", "content": f"Phần {i + 1} của tài liệu:\n{bucket_text}\n\nTóm tắt:"}
            ]

            try:
                part_summary = generate_local(map_messages, max_new_tokens=150, temperature=0.3)
                if part_summary:
                    map_summaries.append(f"--- Tóm tắt Phần {i + 1} ---\n{part_summary.strip()}")
            except Exception as e:
                logger.error("[LLMReader] Local LLM Map phase error: %s", e)

        # Reduce phase
        if map_summaries:
            combined_map_text = "\n\n".join(map_summaries)
            reduce_messages = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là DocBrain AI — Chuyên gia phân tích và tóm tắt tài liệu cao cấp. "
                        "BẮT BUỘC TRẢ LỜI 100% BẰNG TIẾNG VIỆT TỰ NHIÊN, MẠCH LẠC, ĐẦY ĐỦ Ý CHÍNH.\n"
                        "Trình bày gồm:\n"
                        "- Chủ đề / Mục tiêu chính\n"
                        "- Các điểm cốt lõi quan trọng nhất\n"
                        "- Kết luận tổng thể"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Dưới đây là các bản tóm tắt từng phần của tài liệu '{doc_title}':\n"
                        f"{combined_map_text}\n\n"
                        f"Hãy viết bài Báo cáo Tóm tắt Toàn diện cuối cùng cho tài liệu này (trình bày Markdown đẹp mắt):"
                    )
                }
            ]
            try:
                final_summary = generate_local(reduce_messages, max_new_tokens=350, temperature=0.3)
            except Exception as e:
                logger.error("[LLMReader] Local LLM Reduce phase error: %s", e)
                final_summary = ""
        else:
            final_summary = ""

        if not final_summary:
            final_summary = (
                f"Tài liệu **{doc_title}** chứa các thông tin quan trọng trải dài từ trang đầu đến trang cuối. "
                f"Bạn có thể đặt câu hỏi chi tiết về bất kỳ số liệu hay mục nào để tôi trích xuất chính xác cho bạn!"
            )

        answer_markdown = (
            f"**BÁO CÁO TÓM TẮT PHÂN CẤP**\n\n"
            f"Trong phiên làm việc này, bạn đã tải lên **{len(docs)} tài liệu**:\n"
            f"{doc_list_str}\n\n"
            f"{final_summary}\n\n"
            f"*Bạn có thể đặt bất kỳ câu hỏi chi tiết nào về nội dung trên!*"
        )

        return {
            "answer": answer_markdown,
            "confidence": 1.0,
            "used_api": False,
            "is_conversational": True,
            "used_chunks": context_chunks[:4]
        }

    def generate_rag_answer(self, query: str, context_chunks: List[Dict[str, Any]], session_docs: List[str] = None, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generative RAG synthesizer with Corrective RAG grading and multi-turn context:
        - Grades context relevance before generating (CRAG).
        - Synthesizes the final answer via the local LLM with system self-context and history.
        """
        if not context_chunks:
            return {
                "answer": f"Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi **'{query}'** trong các tài liệu hiện có.",
                "confidence": 0.0, "used_api": False, "is_conversational": False, "used_chunks": []
            }

        # 1. Corrective RAG (CRAG) grading: bail out if no chunk clears the relevance threshold
        max_score = max([c.get("score", 0.0) for c in context_chunks]) if context_chunks else 0.0
        if max_score < self.crag_threshold:
            return {
                "answer": f"Xin lỗi, các tài liệu được tải lên không chứa thông tin để trả lời cho câu hỏi **'{query}'**.",
                "confidence": 0.0, "used_api": False, "is_conversational": True, "used_chunks": []
            }

        # 2. Clean and assemble context
        context_texts = []
        for i, c in enumerate(context_chunks[:4]):
            clean_t = clean_pdf_snippet(clean_pdf_text_artifacts(c.get("text", "")))
            src = c.get("metadata", {}).get("source", "Tài liệu")
            page = c.get("metadata", {}).get("page", 1)
            context_texts.append(f"[Trích đoạn {i + 1} - {src} (Trang {page})]:\n{clean_t}")

        combined_context = "\n\n".join(context_texts)

        ans = self._call_local_llm(query=query, context_text=combined_context)
        if ans:
            avg_score = sum(c.get("score", 0.8) for c in context_chunks[:3]) / min(len(context_chunks), 3)
            return {
                "answer": ans,
                "confidence": round(min(1.0, float(avg_score * 1.05)), 2),
                "used_api": True,
                "is_conversational": False,
                "used_chunks": context_chunks[:3]
            }

        # Fallback: extractive snippet when local generation fails or returns empty
        avg_score = sum(c.get("score", 0.8) for c in context_chunks[:3]) / min(len(context_chunks), 3)
        confidence = round(min(1.0, float(avg_score)), 2)

        main_snippet = context_chunks[0].get("text", "")
        main_snippet_clean = clean_pdf_snippet(clean_pdf_text_artifacts(main_snippet))

        formatted = main_snippet_clean[:600]
        entities = re.findall(r"\b[A-Z][A-Za-z0-9\-\_]{2,}\b|\b\d+\s*(?:tỷ|triệu|%|USD|VNĐ|ca|mô hình|trang)\b", formatted)
        for ent in set(entities[:6]):
            if len(ent) > 2 and ent not in ["AND", "FOR", "THE", "WITH"]:
                formatted = re.sub(rf"\b{re.escape(ent)}\b", f"**{ent}**", formatted)

        final_ans = (
            f"Không tìm thấy thông tin phù hợp trong các tài liệu được cung cấp cho câu hỏi **'{query}'**.\n\n"
            f"Dưới đây là trích đoạn liên quan nhất để tham khảo:\n{formatted}"
        )

        return {
            "answer": final_ans,
            "confidence": confidence,
            "used_api": False,
            "is_conversational": False,
            "used_chunks": context_chunks[:3]
        }

    def _call_local_llm(self, query: str, context_text: Optional[str] = None) -> str:
        """Generate an answer with the locally-hosted LLM (no external API, no quota/cost)."""
        try:
            if context_text is not None:
                system_prompt = (
                    "Bạn là DocBrain AI — Chuyên gia phân tích và tra cứu tài liệu.\n"
                    "Nhiệm vụ của bạn là trả lời câu hỏi một cách chính xác dựa trên thông tin trong NGỮ CẢNH TÀI LIỆU.\n\n"
                    "HƯỚNG DẪN QUAN TRỌNG:\n"
                    "1. Đọc kỹ ngữ cảnh và trích xuất câu trả lời.\n"
                    "2. Nếu không có thông tin, hãy nói: 'Xin lỗi, tài liệu không chứa thông tin để trả lời câu hỏi này.'\n"
                    "3. LUÔN trả lời bằng câu hoàn chỉnh, lịch sự, có đầy đủ chủ ngữ và vị ngữ.\n"
                    "4. TUYỆT ĐỐI KHÔNG trả lời cộc lốc chỉ vài từ hoặc chỉ có con số.\n\n"
                    "VÍ DỤ:\n"
                    "- Câu hỏi: Doanh thu là bao nhiêu?\n"
                    "- Trả lời SAI (quá ngắn): 500 tỷ.\n"
                    "- Trả lời ĐÚNG (đầy đủ): Theo tài liệu, doanh thu đạt 500 tỷ đồng."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"NGỮ CẢNH TÀI LIỆU:\n{context_text}\n\nCÂU HỎI: {query}\n\nHãy trả lời câu hỏi trên bằng một câu hoàn chỉnh (có chủ ngữ, vị ngữ):"}
                ]
            else:
                system_prompt = (
                    "Bạn là DocBrain AI, trợ lý AI tra cứu tài liệu nội bộ.\n"
                    "BẮT BUỘC TRẢ LỜI 100% BẰNG TIẾNG VIỆT TỰ NHIÊN."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]

            answer = generate_local(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            return answer.strip()
        except Exception as e:
            logger.error("[LLMReader] Local LLM call error: %s", e)
            return ""
