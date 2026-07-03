from pathlib import Path

from docx import Document
from pypdf import PdfReader

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}





def allowed_extension(extension: str) -> bool:
    return extension.lower() in ALLOWED_EXTENSIONS


def extract_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    return "".join(
        page.extract_text() + "\n"
        for page in reader.pages
        if page.extract_text()
    )


def extract_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def extract_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def extract_text_by_type(file_path: Path, extension: str) -> str:
    ext = extension.lower()
    if ext == "pdf":
        return extract_pdf(file_path)
    if ext == "docx":
        return extract_docx(file_path)
    if ext == "txt":
        return extract_txt(file_path)
    raise ValueError(f"Unsupported file type: {extension}")


