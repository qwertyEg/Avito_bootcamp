import re

import pymorphy3

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
CYRILLIC_RE = re.compile("[а-я]")
SPACES_RE = re.compile(r"\s+")

_morph = None
_lemma_cache = {}


def clean_query(text):
    # приводим запрос к одному виду чтобы одинаковые тексты совпадали
    text = (text or "").lower().replace("ё", "е")
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower().replace("ё", "е"))


def lemma(word):
    global _morph
    result = _lemma_cache.get(word)
    if result is None:
        # цифры и латиницу морфология не знает
        if word.isdigit() or not CYRILLIC_RE.search(word):
            result = word
        else:
            if _morph is None:
                _morph = pymorphy3.MorphAnalyzer()
            result = _morph.parse(word)[0].normal_form
        _lemma_cache[word] = result
    return result


def lemmatize(text):
    return " ".join(lemma(word) for word in tokenize(text))


def lemmatize_docs(docs):
    return [lemmatize(doc) for doc in docs]
