from spellchecker import SpellChecker
import re

spell = SpellChecker()
spell.word_frequency.load_words(["warehouse", "supplies", "logistics", "transport", "Ayodele"])


def normalize_text(text: str) -> str:
    # Allow alphanumerics, arithmetic signs, and common punctuation + \n
    allowed = re.compile(r"[^\w\s\+\-\*xX÷/=^%\n.,!?]")
    text = allowed.sub("", text)
    # Normalize multiple spaces but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)  # Remove leading spaces after newline
    return text.strip()


def clean_ocr_text(raw_text: str) -> str:
    words = raw_text.split()
    corrected_words = []
    for word in words:
        if word.isalpha():
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)
