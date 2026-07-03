from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(
    text: str,
    filename: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    splits = text_splitter.split_text(text)

    return [
        {"filename": filename, "chunk": i, "text": split}
        for i, split in enumerate(splits)
    ]
