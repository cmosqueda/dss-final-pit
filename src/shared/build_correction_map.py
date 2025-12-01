import pandas as pd
import re
import time
import json
from spellchecker import SpellChecker

spellChecker = SpellChecker(language='en')

def build_correction_map(texts: pd.Series, dump_path: str):
    """Build a spelling correction map from a pandas Series and dump as JSON."""
    print("Building spelling correction map...")
    start_time = time.time()

    all_text = ' '.join(texts.astype(str))
    all_text = re.sub(r'[^a-zA-Z]', ' ', all_text).lower()
    unique_words = set(all_text.split())

    misspelled = spellChecker.unknown(unique_words)
    correction_map = {word: spellChecker.correction(word) for word in misspelled}

    # Dump correction map to JSON
    with open(dump_path, 'w') as f:
        json.dump(correction_map, f)

    end_time = time.time()
    print(f"Built correction map in {end_time - start_time:.2f}s")
    print(f"Found {len(misspelled)} typos, mapped {len(correction_map)} corrections.")
    print(f"Correction map saved to {dump_path}")
    return correction_map


