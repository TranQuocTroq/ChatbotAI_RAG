"""
Singleton loader for the local GGUF LLM using llama.cpp for ultra-fast CPU inference.
Replaces the native Transformers loader.
"""
import logging
import threading
from typing import List, Dict, Optional
from src.core.config import get_config

logger = logging.getLogger(__name__)

_model_instance = None
_lock = threading.Lock()

def get_local_llm_model():
    """Return the llama.cpp Llama instance singleton, loading it on first use."""
    global _model_instance
    if _model_instance is None:
        with _lock:
            if _model_instance is None:
                from huggingface_hub import hf_hub_download
                from llama_cpp import Llama

                repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
                filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
                
                logger.info("[LocalLLM-LlamaCPP] Downloading/Verifying GGUF model '%s/%s'...", repo_id, filename)
                try:
                    # Download the model to local cache if not exists
                    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    logger.info("[LocalLLM-LlamaCPP] File ready at: %s", model_path)
                    
                    import os
                    cpu_count = os.cpu_count() or 2
                    optimal_threads = max(1, cpu_count - 1) if cpu_count > 1 else 1

                    # Initialize llama.cpp
                    _model_instance = Llama(
                        model_path=model_path,
                        n_ctx=4096,  # 4K context window to support RAG chunks
                        n_threads=optimal_threads,
                        n_batch=512,
                        verbose=False # Keep standard output clean
                    )
                    logger.info("[LocalLLM-LlamaCPP] Model loaded into memory successfully. (Threads: %d, Batch: 512)", optimal_threads)
                except Exception as e:
                    logger.error("[LocalLLM-LlamaCPP] Failed to load model: %s", e)
                    raise e
                    
    return _model_instance

def generate_local(messages: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """Generate text with the local GGUF Qwen LLM using llama.cpp."""
    model = get_local_llm_model()
    if model is None:
        return ""
        
    try:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Extract text content from OpenAI-style JSON response
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("[LocalLLM-LlamaCPP] Generation error: %s", e)
        return ""
