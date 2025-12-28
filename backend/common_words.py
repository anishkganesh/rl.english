"""Top 10,000 most common English words from Google's word frequency data.

This replaces NLTK's full dictionary which includes obscure/archaic words.
The 10k list is cleaner and covers everyday vocabulary.
Words are ranked by frequency for bonus weighting.
"""

# Valid single-letter English words (rank them as ultra-common)
VALID_SINGLE_LETTERS = {'a': 5, 'i': 8}  # Approximate ranks

# Valid two-letter English words with approximate ranks
VALID_TWO_LETTERS = {
    'of': 2, 'to': 4, 'in': 6, 'is': 8, 'on': 9, 'by': 11, 'it': 16, 
    'or': 18, 'be': 19, 'at': 22, 'as': 23, 'an': 27, 'we': 30, 'us': 33,
    'if': 35, 'my': 37, 'do': 43, 'no': 44, 'he': 60, 'up': 70, 'so': 80,
    'am': 200, 'me': 100, 'go': 150, 'oh': 500, 'hi': 600, 'ok': 700,
    'ox': 5000, 'ax': 5000, 'ow': 5000, 'ah': 3000, 'eh': 4000
}

# Cached data
_common_words = None
_word_ranks = None

def _load_common_words():
    """Load common words list from Google 10k with ranks."""
    import os
    words = set()
    ranks = {}
    
    # Add valid short words with their ranks
    for word, rank in VALID_SINGLE_LETTERS.items():
        words.add(word)
        ranks[word] = rank
    
    for word, rank in VALID_TWO_LETTERS.items():
        words.add(word)
        ranks[word] = rank
    
    # Load from Google 10k file (line number = rank)
    try:
        words_file = os.path.join(os.path.dirname(__file__), 'google10k.txt')
        if os.path.exists(words_file):
            with open(words_file, 'r') as f:
                for rank, line in enumerate(f, start=1):
                    word = line.strip().lower()
                    # Include 3+ letter alphabetic words
                    if len(word) >= 3 and word.isalpha():
                        words.add(word)
                        ranks[word] = rank
    except Exception as e:
        print(f"Error loading words file: {e}")
    
    return words, ranks

def get_common_words():
    """Get set of common English words."""
    global _common_words, _word_ranks
    if _common_words is None:
        _common_words, _word_ranks = _load_common_words()
        print(f"Loaded {len(_common_words)} common English words with ranks")
    return _common_words

def get_word_ranks():
    """Get dictionary mapping words to their frequency rank."""
    global _common_words, _word_ranks
    if _word_ranks is None:
        _common_words, _word_ranks = _load_common_words()
    return _word_ranks

def get_word_weight(word: str) -> float:
    """Get frequency-based weight for a word.
    
    Returns:
        3.0 for rank 1-100 (ultra-common: the, is, you, that)
        2.0 for rank 101-1000 (very common: time, good, work)
        1.0 for rank 1001-10000 (common: rest of vocabulary)
        0.0 if word not in list
    """
    ranks = get_word_ranks()
    rank = ranks.get(word.lower())
    
    if rank is None:
        return 0.0
    elif rank <= 100:
        return 3.0
    elif rank <= 1000:
        return 2.0
    else:
        return 1.0
