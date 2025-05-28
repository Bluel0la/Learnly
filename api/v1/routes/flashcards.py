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
import os, httpx, re, io, asyncio, random
from datetime import datetime

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


# --- Helper: Clean math content and normalize spacing ---
def _clean_math_text(text: str) -> str:
    text = text.replace("×", "*").replace("−", "-").replace("•", "*")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(
        r"(?<=[\w)])\s*\n\s*(?=[\w(])", " ", text
    )  # fix line breaks mid-equation
    return text.strip()


# --- Helper: Remove boilerplate and flatten ---
def _filter_and_clean(lines: list[str], boilerplate: list[str]) -> list[str]:
    return [
        _clean_math_text(line)
        for line in lines
        if line.strip() and not any(bp.lower() in line.lower() for bp in boilerplate)
    ]


# --- Text extractor for slides + notes ---
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


# --- Master extractor function ---
def extract_text_from_file(file: bytes, filename: str) -> str:
    ext = filename.split(".")[-1].lower()

    if ext == "txt":
        return file.decode("utf-8")

    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file))
        pages = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                content = _clean_math_text(content)
                pages.append(content)
        return "\n\n".join(pages)

    elif ext == "docx":
        doc = Document(io.BytesIO(file))
        paragraphs = [
            _clean_math_text(p.text) for p in doc.paragraphs if p.text.strip()
        ]
        return "\n\n".join(paragraphs)

    elif ext == "md":
        return file.decode("utf-8")

    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file))
        slide_texts = []
        notes_texts = []
        for slide in prs.slides:
            # Slide content
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text)

            # Speaker notes
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                for shape in notes_slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        notes_texts.append(shape.text)

        return clean_and_structure_text(slide_texts, notes_texts)

    else:
        raise ValueError(
            "Unsupported file type. Allowed: .txt, .pdf, .docx, .md, .pptx"
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

    # Step 3: Generate flashcards in parallel


async def generate_flashcards_from_notes(
    deck_id: UUID,
    notes: schemas.NoteChunks,
    num_flashcards: int,
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
        print(f"✅ Summary received, {len(key_points)} key points: {key_points}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    if not key_points:
        raise HTTPException(
            status_code=400, detail="No key points returned from summarization."
        )

    # Step 3: Clean key points
    cleaned_key_points = []
    for point in key_points:
        point = re.sub(
            r"Here are the key educational insights.*?:\s*",
            "",
            point,
            flags=re.IGNORECASE,
        )
        point = re.sub(r"^\d+\.\s*", "", point).strip()
        if point:
            cleaned_key_points.append(point)
    if len(cleaned_key_points) < len(key_points):
        print(f"⚠️ Warning: {len(key_points) - len(cleaned_key_points)} key points filtered out during cleaning")
    print(f"🔹 Cleaned key points: {cleaned_key_points}")

    # Step 4: Limit key points to requested number of flashcards
    cleaned_key_points = cleaned_key_points[:num_flashcards]
    print(f"🔹 Processing up to {num_flashcards} key points: {cleaned_key_points}")

    # Step 5: Check flashcard endpoint availability
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(flashcard_url)  # Adjust if GET isn't supported
        resp.raise_for_status()
        print("✅ Flashcard endpoint is available")
    except Exception as e:
        print(f"⚠️ Warning: Flashcard endpoint health check failed: {type(e).__name__}: {str(e)}")

    # Step 6: Generate flashcards in batches
    async def generate_card(point: str, index: int, semaphore: asyncio.Semaphore):
        print(f"🔹 Processing key point {index+1}: {point[:200]}...")
        for attempt in range(3):  # Retry up to 3 times
            try:
                async with semaphore:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                        resp = await client.post(flashcard_url, json={"prompt": point})
                    resp.raise_for_status()
                response_data = resp.json()
                print(f"🔸 Full response JSON for point {index+1}: {response_data}")

                question = response_data.get("question", "").strip()
                answer = response_data.get("answer", "").strip()

                if not question or not answer:
                    print(
                        f"❌ Invalid Q/A pair for point {index+1}: Q={question}, A={answer}, Point: {point}"
                    )
                    return None

                if len(question) < 5:
                    print(f"❌ Question too short for point {index+1}: Q={question}, Point: {point}")
                    return None

                print(
                    f"✅ Valid flashcard for point {index+1}:\nQ: {question}\nA: {answer}\n"
                )
                return {"question": question, "answer": answer}

            except httpx.HTTPStatusError as e:
                print(
                    f"❌ HTTP error for point {index+1}, attempt {attempt+1}: {type(e).__name__}: {str(e)}, Status: {e.response.status_code}, Point: {point}"
                )
                if attempt == 2:
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            except httpx.RequestError as e:
                print(
                    f"❌ Network error for point {index+1}, attempt {attempt+1}: {type(e).__name__}: {str(e)}, Point: {point}"
                )
                if attempt == 2:
                    return None
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(
                    f"❌ Unexpected error for point {index+1}, attempt {attempt+1}: {type(e).__name__}: {str(e)}, Response: {locals().get('response_data', 'No response')}, Point: {point}"
                )
                return None
        return None

    # Process key points in batches
    batch_size = 3
    batch_delay = 2  # seconds
    qa_pairs = []
    for i in range(0, len(cleaned_key_points), batch_size):
        batch = cleaned_key_points[i:i + batch_size]
        print(f"🔹 Processing batch {i//batch_size + 1} with {len(batch)} key points")
        results = await asyncio.gather(
            *[
                generate_card(p, i + j, asyncio.Semaphore(2))  # Limit to 2 concurrent requests per batch
                for j, p in enumerate(batch)
            ]
        )
        batch_qa_pairs = [qa for qa in results if qa]
        qa_pairs.extend(batch_qa_pairs)
        print(f"🔹 Batch {i//batch_size + 1} completed: {len(batch_qa_pairs)} flashcards generated")
        if i + batch_size < len(cleaned_key_points):
            print(f"🔹 Waiting {batch_delay} seconds before next batch")
            await asyncio.sleep(batch_delay)

    if not qa_pairs:
        raise HTTPException(
            status_code=400,
            detail=f"No valid flashcards generated for {num_flashcards} requested."
        )
    if len(qa_pairs) < num_flashcards:
        print(
            f"⚠️ Warning: Generated {len(qa_pairs)} flashcards, fewer than {num_flashcards} requested"
        )

    # Step 7: Save flashcards to database
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
        "message": f"{len(flashcards_to_save)} of {num_flashcards} requested flashcards generated and saved.",
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
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        chunks = [
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
                notes_slide = slide.notes_slide
                notes_text = []
                for shape in notes_slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        notes_text.append(shape.text.strip())
                slide_texts.append("\n".join(notes_text))
            if slide_texts:
                chunks.append("\n".join(slide_texts).strip())
    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        chunks = [
                paragraph.text.strip()
                for paragraph in doc.paragraphs
                if paragraph.text and paragraph.text.strip()
            ]
    elif ext == "txt":
        chunks = [
                line.strip()
                for line in full_text.splitlines()
                if line.strip()
            ]
    else:
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


@flashcards.get("/decks/{deck_id}/practice")
def get_random_flashcard(
    deck_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .all()
    )
    if not cards:
        raise HTTPException(status_code=404, detail="No flashcards found.")

    selected = random.choice(cards)
    return {
        "card_id": selected.card_id,
        "question": selected.card_with_answer.split("\n")[0][3:].strip(),  # after "Q: "
    }


@flashcards.get("/cards/{card_id}/reveal")
def reveal_flashcard_answer(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    return {
        "answer": card.card_with_answer.split("\n")[1][3:].strip(),  # after "A: "
    }


@flashcards.post("/cards/{card_id}/mark-studied")
def mark_card_as_studied(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    card.is_studied = True
    card.times_reviewed += 1
    card.last_reviewed = datetime.utcnow()
    db.commit()

    return {"message": "Flashcard marked as studied."}


@flashcards.post("/cards/{card_id}/bookmark")
def toggle_bookmark_card(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    card.is_bookmarked = not card.is_bookmarked
    db.commit()
    return {
        "message": f"Flashcard {'bookmarked' if card.is_bookmarked else 'unbookmarked'}."
    }


@flashcards.post("/cards/{card_id}/submit-response")
def submit_flashcard_response(
    card_id: UUID,
    review: schemas.FlashcardReviewInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    # Track response
    card.times_reviewed += 1
    card.last_reviewed = datetime.utcnow()
    if review.is_correct:
        card.correct_count += 1
    else:
        card.wrong_count += 1

    db.commit()

    return {
        "message": "Response recorded.",
        "card_id": str(card.card_id),
        "correct": card.correct_count,
        "wrong": card.wrong_count,
        "times_reviewed": card.times_reviewed,
    }


@flashcards.get("/decks/{deck_id}/analytics")
def get_deck_analytics(
    deck_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .all()
    )
    if not cards:
        raise HTTPException(
            status_code=404, detail="No flashcards found for this deck."
        )

    total_cards = len(cards)
    studied_cards = sum(1 for c in cards if c.is_studied)
    total_reviews = sum(c.times_reviewed for c in cards)
    correct_answers = sum(c.correct_count for c in cards)
    wrong_answers = sum(c.wrong_count for c in cards)
    total_attempts = correct_answers + wrong_answers
    accuracy_percent = (
        (correct_answers / total_attempts) * 100 if total_attempts else 0.0
    )

    return {
        "deck_id": str(deck_id),
        "total_cards": total_cards,
        "studied_cards": studied_cards,
        "total_reviews": total_reviews,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "accuracy_percent": round(accuracy_percent, 2),
    }


@flashcards.get("/decks/{deck_id}/quiz", response_model=schemas.QuizStartResponse)
def start_quiz(
    deck_id: UUID,
    num_questions: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .all()
    )
    if not cards:
        raise HTTPException(status_code=404, detail="No flashcards found.")

    if len(cards) < num_questions:
        num_questions = len(cards)

    selected = random.sample(cards, num_questions)

    quiz_cards = []
    for card in selected:
        question = card.card_with_answer.split("\n")[0][3:].strip()  # after 'Q: '
        quiz_cards.append(schemas.QuizCard(card_id=card.card_id, question=question))

    return {"deck_id": deck_id, "cards": quiz_cards}


@flashcards.post("/quiz/submit", response_model=schemas.QuizResults)
def submit_quiz(
    answers: List[schemas.QuizSubmission],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    correct = 0
    wrong = 0
    detailed = []

    for submission in answers:
        card = (
            db.query(card_models.DeckCard)
            .filter_by(card_id=submission.card_id, user_id=current_user.user_id)
            .first()
        )
        if not card:
            detailed.append(
                {
                    "card_id": submission.card_id,
                    "correct": False,
                    "error": "Card not found",
                }
            )
            wrong += 1
            continue

        # Get correct answer from card
        try:
            correct_answer = card.card_with_answer.split("\n")[1][
                3:
            ].strip()  # after 'A: '
        except IndexError:
            correct_answer = ""

        is_correct = submission.user_answer.strip().lower() == correct_answer.lower()

        # Update tracking
        card.times_reviewed += 1
        if is_correct:
            card.correct_count += 1
            correct += 1
        else:
            card.wrong_count += 1
            wrong += 1

        card.last_reviewed = datetime.utcnow()

        detailed.append(
            {
                "card_id": str(card.card_id),
                "your_answer": submission.user_answer,
                "correct_answer": correct_answer,
                "correct": is_correct,
            }
        )

    db.commit()

    return {
        "total_questions": len(answers),
        "correct": correct,
        "wrong": wrong,
        "detailed_results": detailed,
    }
