import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.config import BASE_DIR
from app.services.extractor import allowed_extension

UPLOAD_FOLDER = BASE_DIR / "uploads"
MANIFEST_PATH = BASE_DIR / "ingest_manifest.json"


def _load_manifest() -> Dict[str, Dict[str, float | int]]:
    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_manifest(manifest: Dict[str, Dict[str, float | int]]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _file_signature(file_path: Path) -> Dict[str, float | int]:
    stat = file_path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _needs_processing(filename: str, file_path: Path, manifest: Dict[str, Dict[str, float | int]]) -> bool:
    signature = _file_signature(file_path)
    previous = manifest.get(filename)
    return previous != signature


def record_indexed_file(filename: str, file_path: Path) -> None:
    manifest = _load_manifest()
    manifest[filename] = _file_signature(file_path)
    _save_manifest(manifest)


def ingest_new_uploads(
    index_file: Callable[[Path, str], dict],
    vector_db,
) -> Dict[str, Any]:
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    manifest = _load_manifest()

    indexed_files: List[Dict[str, Any]] = []
    skipped_files: List[str] = []
    errors: List[Dict[str, str]] = []

    for file_path in sorted(UPLOAD_FOLDER.iterdir()):
        if not file_path.is_file():
            continue

        filename = file_path.name
        if "." not in filename:
            skipped_files.append(filename)
            continue

        ext = filename.rsplit(".", 1)[1].lower()
        if not allowed_extension(ext):
            skipped_files.append(filename)
            continue

        if not _needs_processing(filename, file_path, manifest):
            skipped_files.append(filename)
            continue

        try:
            if filename in manifest or filename in vector_db.get_indexed_source_files():
                vector_db.delete_chunks_for_file(filename)

            result = index_file(file_path, filename)
            manifest[filename] = _file_signature(file_path)
            indexed_files.append(result)
        except Exception as exc:
            errors.append({"filename": filename, "error": str(exc)})

    _save_manifest(manifest)

    return {
        "message": (
            f"Ingested {len(indexed_files)} new or updated file(s)."
            if indexed_files
            else "No new or updated files to ingest."
        ),
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
        "indexed_chunks_in_db": vector_db.count(),
        "errors": errors,
    }
