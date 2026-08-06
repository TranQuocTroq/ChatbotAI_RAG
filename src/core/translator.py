"""
CrossLingualTranslator - Cross-lingual helper for the RAG pipeline.

Responsibilities:
1. Detects the document language (English / Vietnamese).
2. Detects the query language (Vietnamese / English).
3. Translates the query into the document language before BM25/FAISS search.
4. Translation runs on the local LLM only - no external inference API.
"""
import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


def remove_accents(input_str: str) -> str:
    """Removes Vietnamese diacritics from a string."""
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


class CrossLingualTranslator:
    """Cross-lingual helper for the RAG pipeline using local LLM translation."""

    def __init__(self):
        pass

    def detect_language(self, text: str) -> str:
        """Detect the language of a string: 'vi' (Vietnamese) or 'en' (English)."""
        if not text:
            return "vi"

        # Presence of Vietnamese diacritics is a strong signal for 'vi'
        if re.search(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", text, re.IGNORECASE):
            return "vi"

        # Check the ratio of common unaccented Vietnamese words
        text_no_accents = remove_accents(text.lower())
        words = set(re.findall(r"\w+", text_no_accents))
        vi_words = {"toi", "ban", "cua", "trong", "co", "la", "khi", "nhung", "cho", "ve", "nay", "dau", "nao", "giai", "quyêt", "du", "lieu"}

        if len(words.intersection(vi_words)) >= 2:
            return "vi"

        return "en"

    def translate_query(self, query: str, target_lang: str) -> str:
        """
        [DEPRECATED] Translate query into target_lang ('en' or 'vi') using the local LLM.
        This method is deprecated due to high latency overhead on CPU. The embedding model
        (multilingual-e5-small) inherently handles cross-lingual queries.
        """
        if not query.strip():
            return query

        lang_name = "English" if target_lang == "en" else "Vietnamese"
        prompt = (
            f"Translate the following sentence to {lang_name}. "
            f"ONLY return the translation, no explanations, no extra characters:\n"
            f"\"{query}\""
        )

        translated = self._call_llm(prompt)
        return translated.strip() if translated else query

    def translate_query_vi_to_en(self, query_vi: str) -> str:
        """[DEPRECATED] Backward-compatibility wrapper around translate_query."""
        return self.translate_query(query_vi, target_lang="en")

    def _call_llm(self, prompt: str) -> str:
        try:
            from src.core.local_llm_engine import generate_local
            messages = [{"role": "user", "content": prompt}]
            return generate_local(messages, max_new_tokens=64, temperature=0.0)
        except Exception as e:
            logger.error("[Translator] LLM translation call error: %s", e)
            return ""
