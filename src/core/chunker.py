from typing import List, Dict, Any
from src.core.config import get_config

class Chunker:
    """Splits text into chunks with a configurable size and overlap."""

    def __init__(self, chunk_size: int = 600, overlap: int = 120, separators: List[str] = None):
        cfg = get_config()
        self.chunk_size = chunk_size or cfg.get("chunking.chunk_size", 600)
        self.overlap = overlap or cfg.get("chunking.overlap", 120)
        self.separators = separators or cfg.get("chunking.separators", ["\n\n", "\n", ". ", "; ", " ", ""])

    def split_pages_into_chunks(self, pages_data: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Takes a list of document pages and produces a list of chunks with full metadata.
        """
        all_chunks = []
        global_chunk_id = 0

        for item in pages_data:
            text = item.get("text", "")
            source = item.get("source", "")
            page = item.get("page", 1)

            if not text.strip():
                continue

            split_texts = self._recursive_split(text, self.separators)

            for idx, chunk_text in enumerate(split_texts):
                if len(chunk_text.strip()) < 30:
                    continue  # Skip chunks that are too short or empty

                global_chunk_id += 1
                all_chunks.append({
                    "chunk_id": f"{session_id}_{source}_p{page}_c{idx}_{global_chunk_id}",
                    "text": chunk_text.strip(),
                    "metadata": {
                        "session_id": session_id,
                        "source": source,
                        "page": page,
                        "chunk_index": idx,
                        "length": len(chunk_text.strip())
                    }
                })

        return all_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        if len(text) <= self.chunk_size:
            return [text]

        # Find the most suitable separator
        sep = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                sep = ""
                break
            if s in text:
                sep = s
                new_separators = separators[i + 1:]
                break

        if sep != "":
            splits = text.split(sep)
        else:
            splits = list(text)

        current_doc = []
        total_len = 0

        for piece in splits:
            piece_len = len(piece) + (len(sep) if sep != "" and current_doc else 0)
            if total_len + piece_len > self.chunk_size and current_doc:
                doc_text = sep.join(current_doc) if sep != "" else "".join(current_doc)
                
                if len(doc_text) > self.chunk_size and new_separators:
                    final_chunks.extend(self._recursive_split(doc_text, new_separators))
                else:
                    final_chunks.append(doc_text[:self.chunk_size])
                
                # Compute the overlap window
                while current_doc:
                    temp_len = sum(len(p) for p in current_doc) + (len(sep) * (len(current_doc) - 1) if sep != "" and len(current_doc) > 0 else 0)
                    if temp_len <= self.overlap:
                        break
                    current_doc.pop(0)
                total_len = sum(len(p) for p in current_doc) + (len(sep) * (len(current_doc) - 1) if sep != "" and len(current_doc) > 0 else 0)
                piece_len = len(piece) + (len(sep) if sep != "" and current_doc else 0)

            current_doc.append(piece)
            total_len += piece_len

        if current_doc:
            doc_text = sep.join(current_doc) if sep != "" else "".join(current_doc)
            if len(doc_text) > self.chunk_size and new_separators:
                final_chunks.extend(self._recursive_split(doc_text, new_separators))
            else:
                final_chunks.append(doc_text[:self.chunk_size])

        return final_chunks
