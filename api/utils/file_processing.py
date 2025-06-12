import os
import io
import re
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
from fastapi import HTTPException


# === 🔧 Heuristic Flashcard Estimation ===
def estimate_flashcard_count(
    ext: str,
    file_bytes: bytes,
    min_per_unit: int = 3,
    min_cards: int = 5,
    max_cards: int = 50,
) -> int:
    try:
        ext = ext.lower()
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            est = len(reader.pages) * min_per_unit

        elif ext == "pptx":
            prs = Presentation(io.BytesIO(file_bytes))
            est = len(prs.slides) * min_per_unit

        elif ext == "docx":
            doc = Document(io.BytesIO(file_bytes))
            est = (
                len([p for p in doc.paragraphs if p.text.strip()]) // 2
            ) * min_per_unit

        elif ext in {"txt", "md"}:
            lines = file_bytes.decode("utf-8", errors="ignore").splitlines()
            est = (len([l for l in lines if l.strip()]) // 5) * min_per_unit

        else:
            est = min_cards  # Fallback estimate for unsupported types

        return max(min_cards, min(est, max_cards))

    except Exception:
        return min_cards


# === 🧼 Text Cleaning ===
def _clean_math_text(text: str) -> str:
    text = text.replace("×", "*").replace("−", "-").replace("•", "*")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(?<=[\w)])\s*\n\s*(?=[\w(])", " ", text)

    # Better math detection
    if re.search(
        r"\d\s*[\+\-×÷*/^=]\s*\d|\bpi\b|\bsqrt\b|\d{1,3}(,\d{3})+", text, re.IGNORECASE
    ):
        text = "[MATH] " + text

    return text.strip()


def _filter_and_clean(lines: list[str], boilerplate: list[str]) -> list[str]:
    return [
        _clean_math_text(line)
        for line in lines
        if line.strip() and not any(bp.lower() in line.lower() for bp in boilerplate)
    ]


# === 🧾 Slide + Notes Structuring ===
def clean_and_structure_text(slide_texts: list[str], notes_texts: list[str]) -> str:
    boilerplate_phrases = ["Click to add text", "Click to add title"]
    slides = _filter_and_clean(slide_texts, boilerplate_phrases)
    notes = _filter_and_clean(notes_texts, boilerplate_phrases)

    sections = []
    if slides:
        sections.append("[Slide Content]\n" + "\n\n".join(slides))
    if notes:
        sections.append("[Speaker Notes]\n" + "\n\n".join(notes))

    return "\n\n".join(sections).strip()


# === 📥 Extraction Logic ===
def extract_text_by_filetype(ext: str, file_bytes: bytes) -> str:
    ext = ext.lower()

    if ext in {"txt", "md"}:
        return file_bytes.decode("utf-8", errors="ignore")

    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(
            [
                _clean_math_text(p.extract_text() or "")
                for p in reader.pages
                if p.extract_text()
            ]
        )

    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n\n".join(
            [_clean_math_text(p.text) for p in doc.paragraphs if p.text.strip()]
        )

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file_bytes))
        slide_texts, notes_texts = [], []

        for slide in prs.slides:
            for shape in slide.shapes:
                text = getattr(shape, "text", "").strip()
                if text:
                    slide_texts.append(text)
            if slide.has_notes_slide:
                for shape in slide.notes_slide.shapes:
                    note = getattr(shape, "text", "").strip()
                    if note:
                        notes_texts.append(note)

        return clean_and_structure_text(slide_texts, notes_texts)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# === 🧩 Chunking Logic ===
def chunk_text_by_type(ext: str, file_bytes: bytes, full_text: str) -> list[str]:
    ext = ext.lower()

    if ext == "pdf":
        return [
            _clean_math_text(p.extract_text() or "")
            for p in PdfReader(io.BytesIO(file_bytes)).pages
            if p.extract_text()
        ]

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file_bytes))
        chunks = []

        for slide in prs.slides:
            slide_parts = []
            for shape in slide.shapes:
                text = getattr(shape, "text", "").strip()
                if text:
                    slide_parts.append(text)

            if slide.has_notes_slide:
                notes_parts = [
                    getattr(shape, "text", "").strip()
                    for shape in slide.notes_slide.shapes
                    if getattr(shape, "text", "").strip()
                ]
                slide_parts.extend(notes_parts)

            if slide_parts:
                chunks.append("\n".join(slide_parts))

        return chunks

    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return [_clean_math_text(p.text) for p in doc.paragraphs if p.text.strip()]

    elif ext in {"txt", "md"}:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        return ["\n".join(lines[i : i + 5]) for i in range(0, len(lines), 5)]

    # Fallback: chunk by paragraphs
    return [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]


# === 🔄 Entry Point ===
def process_file(file_bytes: bytes, filename: str) -> list[str]:
    ext = os.path.splitext(filename)[-1].lower().replace(".", "")
    full_text = extract_text_by_filetype(ext, file_bytes)

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Empty or unsupported file.")

    return chunk_text_by_type(ext, file_bytes, full_text)
