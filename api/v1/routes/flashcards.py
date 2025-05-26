from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from api.utils.authentication import get_current_user
from api.v1.models import deck_card as card_models
from api.v1.schemas import flashcards as schemas
from api.v1.models import deck as models
from api.v1.models.user import User
from sqlalchemy.orm import Session
from api.db.database import get_db
from pptx import Presentation
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Optional
from uuid import UUID
import os, httpx, re, io


flashcards = APIRouter(prefix="/flashcard", tags=["Flashcards"])
model_util_endpoint = os.getenv("MODEL_UTILITY")
model_chat_endpoint = os.getenv("MODEL_ENDPOINT")

# 🔧 Helper: Parse LLM response
def parse_flashcard_response(text: str):
    q_match = re.search(r"Question:\s*(.*)", text, re.IGNORECASE)
    a_match = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE)
    return (
        q_match.group(1).strip() if q_match else "N/A",
        a_match.group(1).strip() if a_match else "N/A",
    )


def clean_and_structure_text(slide_texts: list[str], notes_texts: list[str]) -> str:
    clean_slides = [s.strip() for s in slide_texts if s.strip()]
    clean_notes = [n.strip() for n in notes_texts if n.strip()]

    # Remove boilerplate phrases (customize if needed)
    boilerplate_phrases = ["Click to add text", "Click to add title"]
    clean_slides = [
        line
        for line in clean_slides
        if all(bp not in line for bp in boilerplate_phrases)
    ]
    clean_notes = [
        line
        for line in clean_notes
        if all(bp not in line for bp in boilerplate_phrases)
    ]

    text = ""
    if clean_slides:
        text += "\n\n[Slide Content]\n" + "\n\n".join(clean_slides)
    if clean_notes:
        text += "\n\n[Speaker Notes]\n" + "\n\n".join(clean_notes)

    # Normalize multiple newlines to two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_file(file: bytes, filename: str) -> str:
    ext = filename.split(".")[-1].lower()

    if ext == "txt":
        return file.decode("utf-8")

    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == "docx":
        doc = Document(io.BytesIO(file))
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == "md":
        return file.decode("utf-8")

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file))
        slide_texts = []
        notes_texts = []
        for slide in prs.slides:
            # Extract slide shapes text
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())

            # Extract speaker notes if present
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                notes_text = []
                for shape in notes_slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        notes_text.append(shape.text.strip())
                notes_texts.append("\n".join(notes_text))

        return clean_and_structure_text(slide_texts, notes_texts)

    else:
        raise ValueError(
            "Unsupported file type. Only .txt, .pdf, .docx, .md, .pptx allowed."
        )


# ✅ Create a deck
@flashcards.post("/decks/", response_model=schemas.DeckOut)
def create_deck(
    deck: schemas.DeckCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_deck = models.Deck(title=deck.title, user_id=current_user.user_id)
    db.add(db_deck)
    db.commit()
    db.refresh(db_deck)
    return db_deck


# ✅ Get all decks for the current user
@flashcards.get("/decks/", response_model=List[schemas.DeckOut])
def get_user_decks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(models.Deck).filter_by(user_id=current_user.user_id).all()


# ✅ Add flashcards to a specific deck
@flashcards.post("/decks/{deck_id}/cards/")
def add_cards_to_deck(
    deck_id: UUID,
    payload: schemas.AddCards,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    cards = [
        card_models.DeckCard(
            deck_id=deck_id,
            user_id=current_user.user_id,
            card_with_answer=card.card_with_answer,
        )
        for card in payload.cards
    ]
    db.add_all(cards)
    db.commit()
    return {"message": f"{len(cards)} cards added successfully."}


# ✅ Get all cards in a specific deck
@flashcards.get("/decks/{deck_id}/cards/", response_model=List[schemas.DeckCardOut])
def get_cards_in_deck(
    deck_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found.")
    return db.query(card_models.DeckCard).filter_by(deck_id=deck_id).all()

async def generate_flashcards_from_notes(
    deck_id: UUID,
    notes: schemas.NoteChunks,  # accepts List[str]
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    flashcard_url = f"{model_chat_endpoint}/flashcard"

    # 1. Get the deck and verify ownership
    db_deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # 2. Summarize notes into key points
    summarization_url = f"{model_util_endpoint}/flashcard-summarization"
    try:
        async with httpx.AsyncClient() as client:
            summary_resp = await client.post(
                summarization_url, json={"chunks": notes.chunks}
            )
        summary_resp.raise_for_status()
        key_points = summary_resp.json().get("points", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    if not key_points:
        raise HTTPException(
            status_code=400, detail="No key points returned from summarization model."
        )

    # 3. Generate flashcards (Q&A) for each key point
    qa_pairs = []
    try:
        async with httpx.AsyncClient() as client:
            for point in key_points:
                flashcard_resp = await client.post(
                    flashcard_url, json={"prompt": point}
                )
                flashcard_resp.raise_for_status()
                raw = flashcard_resp.json()
                raw_text = raw.get("response", "")
                question, answer = parse_flashcard_response(raw_text)
                qa_pairs.append({"question": question, "answer": answer})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Flashcard generation failed: {str(e)}"
        )

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No Q&A pairs returned.")

    # 4. Save flashcards
    flashcards_to_save = [
        card_models.DeckCard(
            deck_id=deck_id,
            user_id=current_user.user_id,
            card_with_answer=f"Q: {qa['question']}\nA: {qa['answer']}",
        )
        for qa in qa_pairs
    ]
    db.add_all(flashcards_to_save)
    db.commit()

    return {
        "message": f"{len(flashcards_to_save)} flashcards generated and added to deck.",
        "cards": qa_pairs,
    }

# ✅ Generate flashcards from note chunks
@flashcards.post("/decks/{deck_id}/generate-flashcards/")
async def upload_notes_and_generate_flashcards(
    deck_id: UUID,
    file: UploadFile = File(...),
    debug: Optional[str] = None,  # "raw" or "chunks"
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    allowed_exts = ["txt", "pdf", "docx", "md", "pptx"]
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        file_bytes = await file.read()
        full_text = extract_text_from_file(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")

    # ✅ Debug mode: raw text preview
    if debug == "raw":
        return {
            "filename": filename,
            "extracted_text": full_text[:5000],  # Trimmed for safety
            "length": len(full_text),
        }

    # ✅ Split into chunks
    chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text found in file.")

    # ✅ Debug mode: chunk preview
    if debug == "chunks":
        return {
            "filename": filename,
            "num_chunks": len(chunks),
            "chunks": chunks[:30],  # Limit preview to first 30 chunks
        }

    # ✅ Proceed to flashcard generation
    return await generate_flashcards_from_notes(
        deck_id=deck_id,
        notes=schemas.NoteChunks(chunks=chunks),
        db=db,
        current_user=current_user,
    )
