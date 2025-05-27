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
import os, httpx, re, io, asyncio


flashcards = APIRouter(prefix="/flashcard", tags=["Flashcards"])
model_util_endpoint = os.getenv("MODEL_UTILITY")
model_chat_endpoint = os.getenv("MODEL_ENDPOINT")


# 🔧 Helper: Parse LLM response
def parse_flashcard_response(text: str):
    text = text.strip()

    # Pre-clean conversational preambles
    text = re.sub(
        r"^.*?(?=\bQuestion\s*:)", "", text, flags=re.IGNORECASE | re.DOTALL
    ).strip()

    # Regex extraction
    question_match = re.search(
        r"Question\s*[:：]\s*(.+?)\s*(?=Answer\s*[:：])",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    answer_match = re.search(
        r"Answer\s*[:：]\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if question_match and answer_match:
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return question, answer

    # Fallback
    lines = text.splitlines()
    if len(lines) == 1:
        sentence_split = re.split(r"[:：]", lines[0], maxsplit=1)
        if len(sentence_split) == 2:
            return sentence_split[0].strip() + "?", sentence_split[1].strip()
        else:
            return "What is this about?", lines[0]

    # More than one line fallback
    first_line = lines[0]
    remaining = " ".join(lines[1:]).strip()
    question = first_line.rstrip(":：.") + "?"
    answer = remaining if remaining else "N/A"
    return question, answer


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
    notes: schemas.NoteChunks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    flashcard_url = f"{model_chat_endpoint}/flashcard"
    summarization_url = f"{model_util_endpoint}/flashcard-summarization"

    # Step 1: Validate deck ownership
    db_deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # Step 2: Summarize note chunks
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            summary_resp = await client.post(
                summarization_url, json={"chunks": notes.chunks}
            )
        summary_resp.raise_for_status()
        key_points = summary_resp.json().get("points", [])

        print("✅ Summary received")
        for i, p in enumerate(key_points):
            print(f"Point {i+1}: {p[:200]}...\n")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    if not key_points:
        raise HTTPException(
            status_code=400, detail="No key points returned from summarization."
        )

    # Step 3: Generate flashcards using parallel calls
    async def generate_card(point: str, index: int):
        print(f"\n🔹 Generating flashcard for point {index+1}")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.post(flashcard_url, json={"prompt": point})
            resp.raise_for_status()

            raw_text = resp.json().get("response", "")
            print(f"🔸 Raw response:\n{raw_text}\n")

            # Direct extraction from Q/A format
            match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", raw_text, re.DOTALL)
            if not match:
                print("❌ Pattern mismatch. Skipping.")
                return None

            question = match.group(1).strip()
            answer = match.group(2).strip()

            print(f"🔹 Final Flashcard:\nQ: {question}\nA: {answer}\n")

            return {"question": question, "answer": answer}
        except Exception as e:
            print(f"❌ Flashcard generation failed: {str(e)}")
            return None

    results = await asyncio.gather(
        *[generate_card(p, i) for i, p in enumerate(key_points)]
    )
    qa_pairs = [qa for qa in results if qa]

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No valid Q&A pairs generated.")

    # Step 4: Save to database
    flashcards_to_save = [
        card_models.DeckCard(
            deck_id=deck_id,
            user_id=current_user.user_id,
            card_with_answer=f"Q: {qa['question']}\nA: {qa['answer']}",
        )
        for qa in qa_pairs
    ]
    db.bulk_save_objects(flashcards_to_save)
    db.commit()

    return {
        "message": f"{len(flashcards_to_save)} flashcards generated and saved.",
        "cards": qa_pairs,
    }


@flashcards.post("/decks/{deck_id}/generate-flashcards/")
async def upload_notes_and_generate_flashcards(
    deck_id: UUID,
    file: UploadFile = File(...),
    debug: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    print("DEBUG: Upload handler called ✅")

    allowed_exts = {"txt", "pdf", "docx", "md", "pptx"}
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    if ext not in allowed_exts:
        print(f"ERROR: Unsupported file type: {ext}")
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        file_bytes = await file.read()
        full_text = extract_text_from_file(file_bytes, filename)

        # Normalize text: strip non-ASCII and normalize spacing
        full_text = re.sub(r"[^\x00-\x7F]+", " ", full_text)
        full_text = re.sub(r"\s{3,}", "\n\n", full_text)
        full_text = full_text.strip()

        print(f"DEBUG: File parsed successfully, length={len(full_text)}")
    except Exception as e:
        print(f"ERROR: File parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")

    if debug == "raw":
        print("DEBUG: Returning raw extracted text preview")
        return {
            "filename": filename,
            "extracted_text": full_text[:5000],
            "length": len(full_text),
        }

    chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
    if not chunks:
        print("ERROR: No valid text chunks found after splitting")
        raise HTTPException(status_code=400, detail="No valid text found in file.")

    print(f"DEBUG: Number of chunks extracted: {len(chunks)}")

    if debug == "chunks":
        print("DEBUG: Returning chunk preview")
        return {
            "filename": filename,
            "num_chunks": len(chunks),
            "chunks": chunks[:30],
        }

    try:
        result = await generate_flashcards_from_notes(
            deck_id=deck_id,
            notes=schemas.NoteChunks(chunks=chunks),
            db=db,
            current_user=current_user,
        )
        print(
            f"DEBUG: Flashcards generated successfully, count={len(result.get('cards', []))}"
        )
        return result
    except Exception as e:
        print(f"ERROR during flashcard generation: {e}")
        raise
