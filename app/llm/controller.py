from typing import Any, Dict

from app.tools.web_search import WebSearchTool


def _format_local_context(db_results: Dict[str, Any]) -> str:
    documents = db_results.get("documents", [[]])[0]
    metadatas = db_results.get("metadatas", [[]])[0]

    if not documents:
        return ""

    formatted_chunks = []
    for doc_text, metadata in zip(documents, metadatas):
        source_file = metadata.get("source_file", "Unknown")
        formatted_chunks.append(f"Source: {source_file}\n\n{doc_text}")

    return "\n\n".join(formatted_chunks)


def gather_hybrid_context(query: str, db_results: Dict[str, Any]) -> str:
    local_context = _format_local_context(db_results)
    web_context = WebSearchTool().search(query)

    sections = []
    if local_context:
        sections.append(f"--- Local Knowledge Base ---\n{local_context}")
    if web_context:
        sections.append(f"--- Live Web Context ---\n{web_context}")

    return "\n\n".join(sections) if sections else "No context available."
