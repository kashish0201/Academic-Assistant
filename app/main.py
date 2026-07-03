import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import asyncio

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from app.auth import require_admin

from app.services.extractor import ALLOWED_EXTENSIONS, allowed_extension, extract_text_by_type
from app.services.chunker import chunk_document
from app.services.embedder import EmbeddingService
from app.services.memory import ConversationMemory
from app.services.vector_db import VectorDBService
from app.services.reranker import RerankerService
from app.services.ingest import ingest_new_uploads, record_indexed_file
from app.llm.generator import LLMGenerator, greeting_response, is_greeting

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVAL_K = 10
FINAL_K = 4

vector_db = VectorDBService()
embedder = EmbeddingService()
reranker = RerankerService()
llm_generator = LLMGenerator()
memory = ConversationMemory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    result = ingest_new_uploads(index_file=index_file, vector_db=vector_db)
    if result["indexed_files"]:
        print(
            f"Auto-ingest: indexed {len(result['indexed_files'])} file(s), "
            f"{result['indexed_chunks_in_db']} total chunks."
        )
    yield


app = FastAPI(title="Academic Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def index_file(file_path: Path, filename: str) -> dict:
    ext = filename.rsplit(".", 1)[1].lower()
    if not allowed_extension(ext):
        raise ValueError(
            f"File type not allowed for '{filename}'. "
            f"Choose from: {sorted(ALLOWED_EXTENSIONS)}"
        )

    extracted_text = extract_text_by_type(file_path, ext)
    chunks = chunk_document(
        text=extracted_text,
        filename=filename,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embedded_chunks = embedder.embed_documents(chunks)
    result = vector_db.store_chunks(embedded_chunks)

    return {
        "filename": filename,
        "chunks_indexed": result.get("inserted_count", len(chunks)),
    }


class QueryRequest(BaseModel):
    session_id: str
    query: str


class AdminVerifyRequest(BaseModel):
    api_key: str


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.post("/admin/verify")
def verify_admin_key(body: AdminVerifyRequest):
    from app.config import ADMIN_API_KEY

    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin access is not configured. Set ADMIN_API_KEY in the environment.",
        )
    if body.api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    return {"ok": True}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_and_extract_file(
    file: UploadFile = File(...),
    _: None = Depends(require_admin),
):
    filename = file.filename
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = filename.rsplit(".", 1)[1].lower()
    if not allowed_extension(ext):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Choose from: {sorted(ALLOWED_EXTENSIONS)}",
        )

    safe_filename = Path(filename).name
    file_path = UPLOAD_FOLDER / safe_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}") from e

    try:
        result = index_file(file_path, safe_filename)
        record_indexed_file(safe_filename, file_path)

        return {
            "message": f"Successfully processed and indexed '{safe_filename}'",
            "total_chunks_saved": result["chunks_indexed"],
            "filename": safe_filename,
            "indexed_chunks_in_db": vector_db.count(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index file: {e}") from e


@app.get("/ingest/status")
def ingest_status(_: None = Depends(require_admin)):
    from app.services.ingest import _load_manifest, _needs_processing

    manifest = _load_manifest()
    pending_files = []

    for file_path in sorted(UPLOAD_FOLDER.iterdir()):
        if not file_path.is_file() or "." not in file_path.name:
            continue
        ext = file_path.name.rsplit(".", 1)[1].lower()
        if not allowed_extension(ext):
            continue
        if _needs_processing(file_path.name, file_path, manifest):
            pending_files.append(file_path.name)

    return {
        "indexed_chunks_in_db": vector_db.count(),
        "indexed_files": sorted(vector_db.get_indexed_source_files()),
        "pending_files": pending_files,
    }


@app.post("/ingest")
async def ingest_uploaded_files(_: None = Depends(require_admin)):
    result = ingest_new_uploads(index_file=index_file, vector_db=vector_db)

    if not result["indexed_files"] and result["errors"]:
        raise HTTPException(
            status_code=500,
            detail={"message": "Ingestion failed", "errors": result["errors"]},
        )

    return result


@app.post("/session/new")
def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.get("/sessions")
def list_sessions():
    return {"sessions": memory.list_sessions()}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    deleted = memory.clear_session(session_id)
    return {"session_id": session_id, "deleted_turns": deleted}


@app.get("/session/{session_id}/history")
def get_session_history(session_id: str, limit: int | None = None):
    return {
        "session_id": session_id,
        "history": memory.get_history(session_id, limit=limit),
    }


@app.post("/ask")
async def user_query(session_id: str = Form(...), query: str = Form(...)):
    try:
        if is_greeting(query):
            async def greeting_stream():
                message = greeting_response()
                memory.add_turn(session_id=session_id, role="user", content=query)
                memory.add_turn(session_id=session_id, role="assistant", content=message)
                yield message

            return StreamingResponse(greeting_stream(), media_type="text/plain")

        if not llm_generator.is_query_safe_and_relevant(query):
            raise HTTPException(
                status_code=400,
                detail="Query contains unsafe or irrelevant content.",
            )

        if vector_db.count() == 0:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The assistant is temporarily unavailable. "
                    "The knowledge base has not been set up yet."
                ),
            )

        history = memory.get_history(session_id, limit=10)

        if history:
            search_query = llm_generator.condense_query(query=query, history=history)
        else:
            search_query = query

        embedded_query = embedder.embed_query(search_query)
        results = vector_db.similarity_search(embedded_query, RETRIEVAL_K)
        results = reranker.rerank(search_query, results, top_k=FINAL_K)

        if not results.get("documents", [[]])[0]:
            raise HTTPException(
                status_code=404,
                detail="No matching documents found in the knowledge base.",
            )

        azure_stream = llm_generator.generated_cited_answers(
            query=query,
            db_results=results,
            history=history,
        )

        async def stream_and_save_wrapper():
            full_response_text = ""
            for chunk in azure_stream:
                full_response_text += chunk
                yield chunk
                await asyncio.sleep(0)

            memory.add_turn(session_id=session_id, role="user", content=query)
            memory.add_turn(session_id=session_id, role="assistant", content=full_response_text)

        return StreamingResponse(stream_and_save_wrapper(), media_type="text/plain")

    except HTTPException as http_err:
        # Forward any standard HTTP exceptions (like 400 or 404) without catching them as 500s
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    # Ensure your module path string accurately matches your directory structure
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)