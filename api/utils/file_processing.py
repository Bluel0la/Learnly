from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
from fastapi import HTTPException
import io, re

# 🔧 Estimate flashcard count based on structure
def estimate_flashcard_count(ext: str, file_bytes: bytes, min_per_unit: int = 3, min_cards: int = 5, max_cards: int = 50) -> int:
    try:
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            est = len(reader.pages) * min_per_unit
        elif ext == "pptx":
            prs = Presentation(io.BytesIO(file_bytes))
            est = len(prs.slides) * min_per_unit
        elif ext == "docx":
            doc = Document(io.BytesIO(file_bytes))
            est = (len([p for p in doc.paragraphs if p.text.strip()]) // 2) * min_per_unit
        elif ext in {"txt", "md"}:
            lines = file_bytes.decode("utf-8").splitlines()
            est = (len([l for l in lines if l.strip()]) // 5) * min_per_unit
        else:
            est = 10

        return max(min_cards, min(est, max_cards))
    except Exception:
        return min_cards


# --- 🔧 Flashcard Parser ---
def parse_flashcard_response(text: str):
    text = text.strip()

    # Remove preambles like "Sure, here's a flashcard:"
    text = re.sub(
        r"^.*?(?=\bQuestion\s*:)", "", text, flags=re.IGNORECASE | re.DOTALL
    ).strip()

    question_match = re.search(
        r"Question\s*[:：]\s*(.+?)\s*(?=Answer\s*[:：])",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    answer_match = re.search(r"Answer\s*[:：]\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if question_match and answer_match:
        return question_match.group(1).strip(), answer_match.group(1).strip()

    # Fallback logic
    lines = text.splitlines()
    if len(lines) == 1:
        parts = re.split(r"[:：]", lines[0], maxsplit=1)
        if len(parts) == 2:
            return parts[0].strip() + "?", parts[1].strip()
        return "What is this about?", lines[0].strip()

    question = lines[0].rstrip(":：.") + "?"
    answer = " ".join(lines[1:]).strip() or "N/A"
    return question, answer


# --- 🧠 Math-aware Text Cleaner ---
def _clean_math_text(text: str) -> str:
    text = text.replace("×", "*").replace("−", "-").replace("•", "*")
    text = re.sub(r"\s{2,}", " ", text)  # Excess spaces
    text = re.sub(
        r"(?<=[\w)])\s*\n\s*(?=[\w(])", " ", text
    )  # Mid-expression line breaks
    if re.search(r"[=+*/^√λπ]", text):  # Optional: Tag math content
        text = "[MATH] " + text
    return text.strip()


# --- 📦 Boilerplate Filter ---
def _filter_and_clean(lines: list[str], boilerplate: list[str]) -> list[str]:
    return [
        _clean_math_text(line)
        for line in lines
        if line.strip() and not any(bp.lower() in line.lower() for bp in boilerplate)
    ]


# --- 🧾 Slide + Notes Structuring ---
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


# --- 🗂️ Master File Extraction Handler ---
def extract_text_from_file(file: bytes, filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    print(f"[Extractor] Processing {filename} (.{ext})")

    if ext == "txt" or ext == "md":
        content = file.decode("utf-8")

    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file))
        pages = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                pages.append(_clean_math_text(content))
        content = "\n\n".join(pages)

    elif ext == "docx":
        doc = Document(io.BytesIO(file))
        paragraphs = [
            _clean_math_text(p.text) for p in doc.paragraphs if p.text.strip()
        ]
        content = "\n\n".join(paragraphs)

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file))
        slide_texts, notes_texts = [], []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text)
            if slide.has_notes_slide:
                for shape in slide.notes_slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        notes_texts.append(shape.text)
        content = clean_and_structure_text(slide_texts, notes_texts)

    else:
        raise ValueError(
            "Unsupported file type. Allowed: .txt, .pdf, .docx, .md, .pptx"
        )

    content = content.strip()
    if not content:
        raise HTTPException(
            status_code=400, detail="File appears to be empty or unsupported."
        )

    return content


def chunk_file_by_type(ext: str, file_bytes: bytes, full_text: str) -> list[str]:
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return [
            page.extract_text().strip()
            for page in reader.pages
            if page.extract_text() and page.extract_text().strip()
        ]

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file_bytes))
        chunks = []
        for slide in prs.slides:
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            if slide.has_notes_slide:
                notes_text = [
                    shape.text.strip()
                    for shape in slide.notes_slide.shapes
                    if hasattr(shape, "text") and shape.text.strip()
                ]
                slide_texts.append("\n".join(notes_text))
            if slide_texts:
                chunks.append("\n".join(slide_texts).strip())
        return chunks

    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    elif ext == "txt":
        return [line.strip() for line in full_text.splitlines() if line.strip()]

    return [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
