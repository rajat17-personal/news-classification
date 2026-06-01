import nltk
from nltk.tokenize import word_tokenize
import re
import pandas as pd
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Source bylines / outlet fingerprints that leak label identity in ISOT.
# "reuters" appears in every real article dateline; "century wire", "21st century wire"
# etc. are specific fake-news sites. Stripping these tests whether models learn
# content-level signals rather than outlet identity.
_SOURCE_TOKENS = re.compile(
    r'\b(reuters|century wire|21st century wire|centurywire|'
    r'washington reuters|london reuters|new york reuters|'
    r'chicago reuters|paris reuters|berlin reuters|'
    r'factbox|image via|via reuters)\b',
    re.IGNORECASE,
)

def clean_text(text: str, strip_sources: bool = False):
    if pd.isna(text):
        return ''
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    if strip_sources:
        text = _SOURCE_TOKENS.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def first_paragraph(text: str):
    if pd.isna(text):
        return ''
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 0:
        return paragraphs[0]
    else:
        tokens = text.split()
        return ' '.join(tokens[:200])

def input_text(row: pd.Series, strategy: str = "full_body") -> str:
    title = '' if pd.isna(row['title']) else str(row['title'])
    text  = '' if pd.isna(row['text'])  else str(row['text'])
    if strategy == "full_body":
        return (title + ' ' + text).strip()
    elif strategy == "headline_para":
        return (title + ' ' + first_paragraph(text)).strip() if text else title
    elif strategy == "headline":
        return title
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Choose from 'full_body', 'headline_para', 'headline'.")
