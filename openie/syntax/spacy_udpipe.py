import spacy

from . import clean_text


def parse(text: str, model: spacy.language.Language, format_: str | None = None):
    text = clean_text(text, format_=format_)
    doc = model(text)
    return doc
