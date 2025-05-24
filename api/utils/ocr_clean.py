from spellchecker import SpellChecker
import re

spell = SpellChecker()
spell.word_frequency.load_words(["warehouse", "supplies", "logistics", "transport", "Ayodele"])

def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
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
