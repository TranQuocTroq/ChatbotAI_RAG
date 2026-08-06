"""
Main FastAPI Application Entry Point for DocBrain RAG.

Provides RESTful API endpoints for session management, document uploading,
question-answering with hybrid retrieval, and citation formatting.
Includes file validation limits, optional API Key authentication for destructive actions,
and non-blocking execution for CPU-heavy RAG operations.
"""
import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import src.core.win_fix
from src.core.rag_pipeline import RAGPipeline
from src.core.session_manager import load_sessions_from_disk, save_sessions_to_disk
from src.core.hf_storage import HFDatasetStorage

# Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocBrain AI RAG App",
    description="Production-ready Agentic RAG API with session isolation and durable storage.",
    version="1.0.4"
)

# Initialize RAG Pipeline orchestrator singleton
pipeline = RAGPipeline()

# Production file validation limits
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit per document for large annual reports
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx", ".txt"}


class AskRequest(BaseModel):
    query: str


def verify_admin_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Minimal Authentication Middleware for destructive management endpoints.
    If ADMIN_API_KEY environment variable is configured, validates X-API-Key request header.
    """
    admin_key = os.getenv("ADMIN_API_KEY", "").strip()
    if admin_key:
        if not x_api_key or x_api_key != admin_key:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing 'X-API-Key' header."
            )


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Renders the main web user interface."""
    template_path = Path("src/frontend/templates/index.html")
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template index.html not found.")
    return template_path.read_text(encoding="utf-8")


@app.get("/api/sessions")
async def get_sessions():
    """Retrieves all active session metadata dictionary."""
    sessions = await run_in_threadpool(load_sessions_from_disk)
    return JSONResponse(content=sessions)


@app.post("/api/sessions/new")
async def create_session():
    """Creates a new isolated user workspace session."""
    sessions = await run_in_threadpool(load_sessions_from_disk)
    new_id = f"session_{uuid.uuid4().hex[:12]}"
    sessions[new_id] = {
        "id": new_id,
        "name": f"Session {len(sessions) + 1}",
        "doc_count": 0,
        "chunks": 0,
        "documents": [],
        "messages": []
    }
    await run_in_threadpool(save_sessions_to_disk, sessions)
    logger.info("Created new session: %s", new_id)
    return JSONResponse(content=sessions[new_id])


@app.delete("/api/sessions/{session_id}", dependencies=[Depends(verify_admin_key)])
async def delete_session(session_id: str):
    """Deletes an entire session and its vector/document stores."""
    sessions = await run_in_threadpool(load_sessions_from_disk)
    if session_id in sessions:
        await run_in_threadpool(pipeline.vector_store.delete_session_store, session_id)
        await run_in_threadpool(pipeline.bm25_store.storage.delete_path, f"sessions_raw/{session_id}")
        
        raw_dir = Path("data/sessions_raw") / session_id
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
            
        del sessions[session_id]
        await run_in_threadpool(save_sessions_to_disk, sessions)
        logger.info("Deleted session: %s", session_id)
        return {"status": "deleted", "session_id": session_id}
        
    raise HTTPException(status_code=404, detail="Session not found.")


@app.post("/api/sessions/{session_id}/upload")
async def upload_file_to_session(session_id: str, file: UploadFile = File(...)):
    """
    Validates, uploads, and ingests a document into a specific session's vector store.
    Enforces file size limit (100MB) and format whitelist (.pdf, .docx, .txt).
    """
    sessions = await run_in_threadpool(load_sessions_from_disk)
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_FILE_EXTENSIONS))}"
        )

    # 2. Validate file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({round(file_size / (1024*1024), 2)}MB) exceeds maximum limit of 100MB."
        )

    # 3. Save to local ephemeral storage and mirror to HF Dataset Repo
    session_raw_dir = Path("data/sessions_raw") / session_id
    session_raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = session_raw_dir / file.filename

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    await run_in_threadpool(
        HFDatasetStorage.get_instance().upload_file,
        target_path,
        f"sessions_raw/{session_id}/{file.filename}"
    )

    # 4. CPU-bound document ingestion and indexing via thread pool
    ingest_res = await run_in_threadpool(
        pipeline.process_and_ingest_file_for_session,
        session_id,
        str(target_path)
    )

    filename = file.filename
    if filename not in sessions[session_id]["documents"]:
        sessions[session_id]["documents"].append(filename)
        sessions[session_id]["doc_count"] = len(sessions[session_id]["documents"])

    sessions[session_id]["chunks"] = ingest_res.get("total_chunks", 0)

    if "Session" in sessions[session_id]["name"] or "No document" in sessions[session_id]["name"]:
        sessions[session_id]["name"] = f"Session: {filename[:15]}"

    await run_in_threadpool(save_sessions_to_disk, sessions)
    logger.info("Successfully ingested document '%s' for session '%s'", filename, session_id)

    return {
        "status": "success",
        "filename": filename,
        "new_chunks": ingest_res.get("new_chunks", 0),
        "total_chunks": sessions[session_id]["chunks"],
        "documents": sessions[session_id]["documents"]
    }


@app.delete("/api/sessions/{session_id}/documents/{filename}", dependencies=[Depends(verify_admin_key)])
async def delete_document_from_session(session_id: str, filename: str):
    """Deletes a specific document from a session and rebuilds vector & BM25 indices."""
    sessions = await run_in_threadpool(load_sessions_from_disk)
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    if filename in sessions[session_id]["documents"]:
        sessions[session_id]["documents"].remove(filename)

    file_path = Path("data/sessions_raw") / session_id / filename
    if file_path.exists():
        os.remove(file_path)
        
    await run_in_threadpool(
        HFDatasetStorage.get_instance().delete_path,
        f"sessions_raw/{session_id}/{filename}"
    )

    # Rebuild indices for remaining documents
    await run_in_threadpool(pipeline.vector_store.delete_session_store, session_id)
    remaining_files = list((Path("data/sessions_raw") / session_id).glob("*"))
    
    total_chunks = 0
    for r_file in remaining_files:
        res = await run_in_threadpool(
            pipeline.process_and_ingest_file_for_session,
            session_id,
            str(r_file)
        )
        total_chunks = res.get("total_chunks", 0)

    sessions[session_id]["doc_count"] = len(sessions[session_id]["documents"])
    sessions[session_id]["chunks"] = total_chunks

    await run_in_threadpool(save_sessions_to_disk, sessions)
    logger.info("Deleted document '%s' from session '%s'", filename, session_id)

    return {
        "status": "deleted",
        "filename": filename,
        "doc_count": sessions[session_id]["doc_count"],
        "chunks": total_chunks,
        "documents": sessions[session_id]["documents"]
    }


@app.post("/api/sessions/{session_id}/ask")
async def ask_question(session_id: str, req: AskRequest):
    """
    Processes user query using non-blocking threadpool for Agentic Intent Routing,
    Hybrid Retrieval (FAISS + BM25 + Reranker), and LLM Generation.
    """
    sessions = await run_in_threadpool(load_sessions_from_disk)
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    doc_cnt = sessions[session_id].get("doc_count", 0)
    session_docs = sessions[session_id].get("documents", [])
    chat_history = sessions[session_id].get("messages", [])
    
    # Execute CPU-bound RAG pipeline in thread pool to keep asyncio event loop responsive
    result = await run_in_threadpool(
        pipeline.ask,
        query=req.query, 
        session_id=session_id, 
        session_doc_count=doc_cnt,
        session_docs=session_docs,
        chat_history=chat_history
    )

    answer = result["answer"]
    exec_time = result["execution_time_sec"]

    is_no_citations = result.get("is_conversational", False) or result["confidence"] == 0.0 or len(result.get("sources", [])) == 0

    if is_no_citations:
        full_bot_response = answer
    else:
        citations_str = "\n\n**Source Citations:**\n"
        for src in result["sources"]:
            snippet = src.get("text_snippet", "").replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            citations_str += f"- **File**: `{src['source_file']}` (Page {src['page']})\n"
            citations_str += f"  > *\"{snippet}\"*\n"
        full_bot_response = f"{answer}{citations_str}"

    if "messages" not in sessions[session_id]:
        sessions[session_id]["messages"] = []

    sessions[session_id]["messages"].append({"role": "user", "content": req.query})
    sessions[session_id]["messages"].append({"role": "assistant", "content": full_bot_response})

    await run_in_threadpool(save_sessions_to_disk, sessions)

    return {
        "query": req.query,
        "answer_short": answer,
        "answer_full": full_bot_response,
        "sources": result["sources"] if not is_no_citations else [],
        "confidence": result["confidence"],
        "execution_time_sec": exec_time
    }
