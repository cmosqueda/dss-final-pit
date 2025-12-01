import pandas as pd
import re
import json
import nltk
from spellchecker import SpellChecker

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# CONTRACTION MAP
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "i've" : "i have",
    "i'd" : "i had",
    "you've" : "you have",
    "you'd" : "you had",
    "we've" : "we have",
    "we'd" : "we had",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "wasn't" : "was not",
    "that's": "that is",
    "you'll" : "you will"
}


spellChecker = SpellChecker(language='en')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# BASIC HELPERS

def clear_null_values(text):
    """Ensure text is string; replace nulls with empty string."""
    return text if isinstance(text, str) else ""


def tokenize_text(text):
    """Tokenize text using NLTK."""
    if not isinstance(text, str):
        return []
    return word_tokenize(text)

def remove_punctuations(text):
    """Remove punctuation and unwanted characters manually."""
    punctuations = '''!()-[]{};:'.'"‘’“”«»„\,<>./?@#$%^–—&_0123456789~*+'''
    cleaned = "".join([char for char in text if char not in punctuations])
    return cleaned

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                         flags=re.IGNORECASE | re.DOTALL)

    def replace(match):
        matched = match.group(0)
        first_char = matched[0]
        expanded = contraction_mapping.get(matched) or contraction_mapping.get(matched.lower())
        expanded = first_char + expanded[1:]
        return expanded

    expanded_text = pattern.sub(replace, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_repeating_characters(tokens):
    repeated_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    substitution = r'\1\2\3'

    def replace(word):
        if wordnet.synsets(word):  
            return word
        new_word = repeated_pattern.sub(substitution, word)
        return replace(new_word) if new_word != word else new_word

    return [replace(word) for word in tokens]


def remove_stopwords(tokens):
    stopWords = set(stopwords.words('english'))
    cleaned = [w for w in tokens if isinstance(w, str) and w not in stopWords and w.strip()]
    return cleaned


def apply_corrections(tokens, correction_map):
    """Apply spelling corrections from a correction map to a list of tokens"""
    return [correction_map.get(word, word) for word in tokens]

# APPLYING CORRECTION MAP
def load_correction_map(json_path: str):
    """Load a correction map from a JSON file."""
    with open(json_path) as f:
        return json.load(f)
    

# FULL CLEANING PIPELINE
def clean_text_pipeline(
    text,
    correction_map=None,
    do_spellcheck=False,
    do_stemming=False,
    do_lemmatization=False
):
    """Complete custom preprocessing pipeline."""
    # Checker
    if not isinstance(text, str) or not text.strip():
        return ""  # return empty string for null/empty input

    # 1 - nulls
    text = clear_null_values(text)

    # 2 - contractions
    text = expand_contractions(text)

    # 3 - punctuation
    text = remove_punctuations(text)

    # 4 - lowercase
    text = text.lower()

    # 5 - tokenize
    tokens = tokenize_text(text)

    # 6 - repeating chars
    tokens = remove_repeating_characters(tokens)

    # 7 - spell check
    if do_spellcheck and correction_map is not None:
        tokens = apply_corrections(tokens, correction_map)

    # 8 - stopwords
    tokens = remove_stopwords(tokens)

    # 9 - stemming
    if do_stemming:
        tokens = [stemmer.stem(word) for word in tokens]

    # 10 - lemmatization
    if do_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 11 - join back
    return " ".join(tokens).strip()


# APPLIED TO DATAFRAME
def preprocess_dataframe(df: pd.DataFrame, text_column: str, label_column: str, correction_map_path: str, do_spellcheck=False, do_stemming=False, do_lemmatization=False):
    """Preprocess an entire dataframe."""
    df = df.copy()

    # Load correction map
    correction_map = load_correction_map(correction_map_path)

    # Apply pipeline row by row
    print("Applying cleaning pipeline to dataset...")
    df[text_column] = df[text_column].apply(
        lambda x: clean_text_pipeline(x, correction_map=correction_map, do_spellcheck=do_spellcheck, do_stemming=do_stemming, do_lemmatization=do_lemmatization)
    )

    if label_column:
        return df[[text_column, label_column]]
    return df[[text_column]]