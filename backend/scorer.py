"""Scoring system using hybrid approach: Perplexity + Relevance + Structure."""

import os
import json
import asyncio
import re
import math
from typing import Dict, List, Tuple, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from config import config
from common_words import get_common_words, get_word_weight
import torch

load_dotenv()


# =============================================================================
# Hybrid Scoring Components
# =============================================================================

class PerplexityScorer:
    """Scores text naturalness using GPT-2 perplexity.
    
    Lower perplexity = more natural English.
    Perplexity < 50: Very natural (score 0.9-1.0)
    Perplexity 50-150: Mostly natural (score 0.6-0.9)
    Perplexity 150-500: Some issues (score 0.3-0.6)
    Perplexity > 500: Word salad (score 0.0-0.3)
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
                print("[PerplexityScorer] Loading distilgpt2 model...")
                self._tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
                self._model = GPT2LMHeadModel.from_pretrained("distilgpt2")
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model.to(self._device)
                self._model.eval()
                print(f"[PerplexityScorer] Model loaded on {self._device}")
            except Exception as e:
                print(f"[PerplexityScorer] Failed to load model: {e}")
                self._model = None
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for the given text.
        
        Returns:
            Perplexity value (lower = more natural). Returns 1000.0 on error.
        """
        if not text or len(text.strip()) < 2:
            return 1000.0
        
        self._ensure_loaded()
        if self._model is None:
            return 500.0  # Fallback if model failed to load
        
        try:
            # Tokenize
            encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(self._device)
            
            if input_ids.size(1) < 2:
                return 1000.0  # Too short
            
            # Calculate loss
            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
            
            perplexity = math.exp(loss)
            return min(perplexity, 10000.0)  # Cap at 10000
            
        except Exception as e:
            print(f"[PerplexityScorer] Error calculating perplexity: {e}")
            return 1000.0
    
    def score(self, text: str) -> Tuple[float, float]:
        """Convert perplexity to a 0-1 score.
        
        Returns:
            Tuple of (score, perplexity)
        """
        perplexity = self.calculate_perplexity(text)
        
        # Convert perplexity to 0-1 score
        if perplexity < 30:
            score = 0.95 + (30 - perplexity) / 30 * 0.05  # 0.95-1.0
        elif perplexity < 50:
            score = 0.90 + (50 - perplexity) / 20 * 0.05  # 0.90-0.95
        elif perplexity < 100:
            score = 0.70 + (100 - perplexity) / 50 * 0.20  # 0.70-0.90
        elif perplexity < 200:
            score = 0.50 + (200 - perplexity) / 100 * 0.20  # 0.50-0.70
        elif perplexity < 500:
            score = 0.25 + (500 - perplexity) / 300 * 0.25  # 0.25-0.50
        elif perplexity < 1000:
            score = 0.10 + (1000 - perplexity) / 500 * 0.15  # 0.10-0.25
        else:
            score = max(0.05, 0.10 - (perplexity - 1000) / 5000 * 0.05)  # 0.05-0.10
        
        return (score, perplexity)


class RelevanceScorer:
    """Scores response relevance using sentence embeddings.
    
    Uses cosine similarity between agent response and GPT context.
    Similarity > 0.5: Highly relevant (score 0.8-1.0)
    Similarity 0.3-0.5: Somewhat relevant (score 0.5-0.8)
    Similarity < 0.3: Irrelevant (score 0.0-0.5)
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[RelevanceScorer] Loading sentence-transformers model...")
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[RelevanceScorer] Model loaded")
            except Exception as e:
                print(f"[RelevanceScorer] Failed to load model: {e}")
                self._model = None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.
        
        Returns:
            Similarity value (0-1). Returns 0.0 on error.
        """
        if not text1 or not text2:
            return 0.0
        
        self._ensure_loaded()
        if self._model is None:
            return 0.3  # Fallback if model failed to load
        
        try:
            # Get embeddings
            embeddings = self._model.encode([text1, text2])
            
            # Calculate cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            
            similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            print(f"[RelevanceScorer] Error calculating similarity: {e}")
            return 0.0
    
    def score(self, agent_text: str, gpt_context: str) -> Tuple[float, float]:
        """Convert similarity to a 0-1 relevance score.
        
        Args:
            agent_text: The agent's response
            gpt_context: GPT's previous message
        
        Returns:
            Tuple of (score, similarity)
        """
        if not gpt_context:
            # No context to compare against - return neutral score
            return (0.5, 0.0)
        
        similarity = self.calculate_similarity(agent_text, gpt_context)
        
        # Convert similarity to 0-1 score
        if similarity > 0.6:
            score = 0.90 + (similarity - 0.6) / 0.4 * 0.10  # 0.90-1.0
        elif similarity > 0.5:
            score = 0.80 + (similarity - 0.5) / 0.1 * 0.10  # 0.80-0.90
        elif similarity > 0.4:
            score = 0.65 + (similarity - 0.4) / 0.1 * 0.15  # 0.65-0.80
        elif similarity > 0.3:
            score = 0.50 + (similarity - 0.3) / 0.1 * 0.15  # 0.50-0.65
        elif similarity > 0.2:
            score = 0.30 + (similarity - 0.2) / 0.1 * 0.20  # 0.30-0.50
        elif similarity > 0.1:
            score = 0.15 + (similarity - 0.1) / 0.1 * 0.15  # 0.15-0.30
        else:
            score = similarity * 1.5  # 0.0-0.15
        
        return (score, similarity)


# Singleton instances
_perplexity_scorer: Optional[PerplexityScorer] = None
_relevance_scorer: Optional[RelevanceScorer] = None


def get_perplexity_scorer() -> PerplexityScorer:
    """Get the singleton PerplexityScorer instance."""
    global _perplexity_scorer
    if _perplexity_scorer is None:
        _perplexity_scorer = PerplexityScorer()
    return _perplexity_scorer


def get_relevance_scorer() -> RelevanceScorer:
    """Get the singleton RelevanceScorer instance."""
    global _relevance_scorer
    if _relevance_scorer is None:
        _relevance_scorer = RelevanceScorer()
    return _relevance_scorer


def score_hybrid(
    text: str,
    gpt_context: Optional[str] = None,
    phase: int = 1
) -> Dict:
    """Score text using hybrid approach: Perplexity + Relevance + Structure.
    
    OPTIMIZED: For Phase 1-2, skip expensive perplexity/relevance scoring.
    Structure scoring is sufficient and much faster.
    
    Args:
        text: The text to score
        gpt_context: GPT's previous message for relevance scoring
        phase: Current curriculum phase (affects structure scoring)
    
    Returns:
        Dict with scoring components
    """
    if not text or not text.strip():
        return {
            "perplexity_score": 0.0,
            "relevance_score": 0.0,
            "structure_score": 0.0,
            "total": 0.0,
            "vocabulary_score": 0.0,
            "grammar_score": 0.0,
            "coherence_score": 0.0,
            "details": {"reason": "Empty text"}
        }
    
    text = text.strip()
    
    # Component 3: Structure Score (fast, local, always computed)
    structure_result = score_structure_enhanced(text, phase, gpt_context)
    structure_score = structure_result["structure_score"]
    
    # OPTIMIZATION: For Phase 1-2, skip expensive ML scoring
    # Structure is the primary signal anyway (70% in Phase 1, 50% in Phase 2)
    if phase <= 2:
        # Fast path: Use structure-based estimates for perplexity/relevance
        # Estimate perplexity from structure quality
        perplexity_score = 0.3 + structure_score * 0.4  # 0.3-0.7 range based on structure
        raw_perplexity = 500 - (structure_score * 400)  # Estimated
        
        # Estimate relevance from word count and patterns
        word_count = structure_result.get("word_count", 0)
        relevance_score = 0.4 + min(0.3, word_count * 0.05)  # 0.4-0.7 based on word count
        raw_similarity = 0.0
        
        if phase == 1:
            # Phase 1: 85% structure, 10% perplexity, 5% relevance (FAST)
            total = structure_score * 0.85 + perplexity_score * 0.10 + relevance_score * 0.05
        else:
            # Phase 2: 70% structure, 20% perplexity, 10% relevance (FAST)
            total = structure_score * 0.70 + perplexity_score * 0.20 + relevance_score * 0.10
    else:
        # Phase 3+: Use full ML scoring (slower but more accurate)
        perplexity_scorer = get_perplexity_scorer()
        perplexity_score, raw_perplexity = perplexity_scorer.score(text)
        
        # Relevance only if we have context
        if gpt_context:
            relevance_scorer = get_relevance_scorer()
            relevance_score, raw_similarity = relevance_scorer.score(text, gpt_context)
        else:
            relevance_score, raw_similarity = 0.5, 0.0
        
        if phase == 3:
            # Phase 3: 30% structure, 35% perplexity, 35% relevance
            total = structure_score * 0.30 + perplexity_score * 0.35 + relevance_score * 0.35
        else:
            # Phase 4+: Standard hybrid (35/35/30)
            total = perplexity_score * 0.35 + relevance_score * 0.35 + structure_score * 0.30
    
    # Map to legacy score names for compatibility
    vocabulary_score = 1.0 if structure_result.get("word_count", 0) > 0 else 0.0
    grammar_score = perplexity_score
    coherence_score = relevance_score
    
    return {
        "perplexity_score": perplexity_score,
        "relevance_score": relevance_score,
        "structure_score": structure_score,
        "total": min(1.0, total),
        # Legacy compatibility
        "vocabulary_score": vocabulary_score,
        "grammar_score": grammar_score,
        "coherence_score": coherence_score,
        # Structure details
        "has_capital_start": structure_result.get("has_capital_start", False),
        "has_punctuation_end": structure_result.get("has_punctuation_end", False),
        "has_proper_spacing": structure_result.get("has_proper_spacing", False),
        "word_count": structure_result.get("word_count", 0),
        "bigram_count": structure_result.get("bigram_count", 0),
        "has_subject_verb": structure_result.get("has_subject_verb", False),
        "has_svo": structure_result.get("has_svo", False),
        "phase_mastery": structure_result.get("phase_mastery", 0.0),
        "discovered_bigrams": structure_result.get("found_bigrams", []),
        "discovered_trigrams": structure_result.get("found_trigrams", []),
        "is_valid_sentence": structure_result.get("has_svo", False),
        # Raw details for debugging
        "details": {
            "raw_perplexity": raw_perplexity,
            "raw_similarity": raw_similarity,
            "reason": f"PPL={raw_perplexity:.1f}, SIM={raw_similarity:.2f}"
        }
    }


def score_structure_enhanced(text: str, phase: int = 1, gpt_context: Optional[str] = None) -> Dict:
    """Enhanced structure scoring with response keyword detection.
    
    Args:
        text: The text to score
        phase: Current curriculum phase
        gpt_context: GPT's previous message for response detection
    
    Returns:
        Dict with structure_score and all component bonuses
    """
    if not text or not text.strip():
        return {
            "structure_score": 0.0,
            "has_capital_start": False,
            "has_punctuation_end": False,
            "has_proper_spacing": False,
            "word_count": 0,
            "bigram_count": 0,
            "has_subject_verb": False,
            "has_svo": False,
            "phase_mastery": 0.0,
            "found_bigrams": [],
            "found_trigrams": [],
            "response_bonus": 0.0
        }
    
    text = text.strip()
    words = [w.strip('.,!?;:').lower() for w in text.split() if w.strip('.,!?;:')]
    
    # === Basic Structure ===
    has_capital_start = text[0].isupper()
    capital_bonus = 0.15 if has_capital_start else 0.0
    
    has_punctuation_end = text[-1] in '.!?'
    punctuation_bonus = 0.15 if has_punctuation_end else 0.0
    
    word_count = len(words)
    has_proper_spacing = 2 <= word_count <= 15  # Extended for longer sentences
    spacing_bonus = 0.10 if has_proper_spacing else (0.05 if word_count > 0 else 0.0)
    
    # === Patterns ===
    found_bigrams, found_trigrams = detect_local_patterns(text)
    bigram_count = len(found_bigrams)
    bigram_bonus = min(0.30, bigram_count * 0.15)
    
    # Subject-verb detection
    has_subject_verb = False
    for i in range(len(words) - 1):
        if words[i] in SUBJECT_PRONOUNS and words[i + 1] in COMMON_VERBS:
            has_subject_verb = True
            break
    subject_verb_bonus = 0.15 if has_subject_verb else 0.0
    
    # SVO detection
    has_svo = False
    svo_bonus = 0.0
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if (words[i] in SUBJECT_PRONOUNS and 
                words[i + 1] in COMMON_VERBS and 
                (words[i + 2] in COMMON_OBJECTS or len(words[i + 2]) > 2)):
                has_svo = True
                svo_bonus = 0.15
                break
    
    # === Response Keywords (NEW) ===
    response_bonus = 0.0
    if gpt_context:
        gpt_lower = gpt_context.lower()
        agent_lower = text.lower()
        first_word = words[0] if words else ""
        
        # Bonus for appropriate response starters
        response_starters = {"yes", "no", "i", "we", "they", "it", "maybe", "sure", "well", "thank", "hello", "hi"}
        if first_word in response_starters:
            response_bonus += 0.05
        
        # Bonus for question response
        if gpt_context.strip().endswith('?'):
            if first_word in {"yes", "no", "i", "we", "maybe", "sure"}:
                response_bonus += 0.05
        
        # Bonus for greeting response
        if any(g in gpt_lower for g in ["hello", "hi", "hey", "how are"]):
            if any(r in agent_lower for r in ["hello", "hi", "hey", "good", "fine", "well", "thank"]):
                response_bonus += 0.05
    
    # === Total based on phase ===
    if phase == 1:
        structure_score = capital_bonus + punctuation_bonus + spacing_bonus
        phase_mastery = 1.0 if (has_capital_start and has_punctuation_end) else 0.0
    elif phase == 2:
        structure_score = capital_bonus + punctuation_bonus + spacing_bonus + bigram_bonus + subject_verb_bonus
        phase_mastery = 1.0 if bigram_count >= 1 else 0.0
    elif phase >= 3:
        structure_score = (capital_bonus + punctuation_bonus + spacing_bonus + 
                         bigram_bonus + subject_verb_bonus + svo_bonus + response_bonus)
        phase_mastery = 1.0 if has_svo else 0.0
    else:
        structure_score = capital_bonus + punctuation_bonus + spacing_bonus
        phase_mastery = 0.0
    
    return {
        "structure_score": min(1.0, structure_score),
        "has_capital_start": has_capital_start,
        "has_punctuation_end": has_punctuation_end,
        "has_proper_spacing": has_proper_spacing,
        "word_count": word_count,
        "bigram_count": bigram_count,
        "has_subject_verb": has_subject_verb,
        "has_svo": has_svo,
        "phase_mastery": phase_mastery,
        "found_bigrams": found_bigrams,
        "found_trigrams": found_trigrams,
        "response_bonus": response_bonus
    }

# Common English bigrams for n-gram pattern bonus
COMMON_BIGRAMS = {
    # Subject-verb patterns
    ("i", "am"): 0.12, ("i", "have"): 0.10, ("i", "will"): 0.08, ("i", "can"): 0.08,
    ("i", "want"): 0.10, ("i", "need"): 0.08, ("i", "like"): 0.08, ("i", "think"): 0.08,
    ("you", "are"): 0.12, ("you", "have"): 0.10, ("you", "can"): 0.08, ("you", "will"): 0.08,
    ("you", "want"): 0.08, ("you", "need"): 0.08, ("you", "should"): 0.08,
    ("he", "is"): 0.10, ("she", "is"): 0.10, ("it", "is"): 0.10, ("it", "was"): 0.08,
    ("we", "are"): 0.10, ("we", "have"): 0.08, ("we", "can"): 0.08, ("we", "need"): 0.08,
    ("they", "are"): 0.10, ("they", "have"): 0.08, ("they", "can"): 0.08,
    # Question patterns
    ("do", "you"): 0.10, ("can", "you"): 0.10, ("will", "you"): 0.08,
    ("do", "i"): 0.08, ("can", "i"): 0.10, ("how", "are"): 0.10, ("how", "do"): 0.08,
    ("what", "is"): 0.10, ("where", "is"): 0.08, ("who", "is"): 0.08,
    ("what", "do"): 0.08, ("what", "are"): 0.08, ("when", "is"): 0.08,
    # Common phrases
    ("there", "is"): 0.08, ("that", "is"): 0.08, ("this", "is"): 0.10,
    ("want", "to"): 0.08, ("have", "to"): 0.08, ("need", "to"): 0.08,
    ("going", "to"): 0.08, ("like", "to"): 0.08, ("able", "to"): 0.08,
    ("would", "like"): 0.08, ("would", "be"): 0.08, ("could", "be"): 0.08,
    # Articles + nouns (very common)
    ("the", "best"): 0.06, ("the", "new"): 0.06, ("the", "first"): 0.06, ("the", "world"): 0.06,
    ("a", "new"): 0.06, ("a", "good"): 0.06, ("a", "great"): 0.06, ("a", "lot"): 0.06,
    ("an", "important"): 0.06, ("an", "interesting"): 0.06,
    # Preposition patterns
    ("in", "the"): 0.06, ("on", "the"): 0.06, ("at", "the"): 0.06, ("to", "the"): 0.06,
    ("of", "the"): 0.06, ("for", "the"): 0.06, ("with", "the"): 0.06,
    ("in", "a"): 0.06, ("on", "a"): 0.06, ("for", "a"): 0.06,
    # Other common pairs
    ("more", "than"): 0.06, ("such", "as"): 0.06, ("as", "well"): 0.06,
    ("each", "other"): 0.06, ("right", "now"): 0.06, ("so", "much"): 0.06,
    # Greetings
    ("hello", "how"): 0.10, ("hi", "how"): 0.10, ("good", "morning"): 0.10,
    ("nice", "to"): 0.08, ("thank", "you"): 0.10, ("yes", "i"): 0.06,
}

# Common trigrams for extra bonus
COMMON_TRIGRAMS = {
    ("i", "am", "a"): 0.15, ("i", "have", "a"): 0.12, ("i", "want", "to"): 0.12,
    ("i", "need", "to"): 0.12, ("i", "like", "to"): 0.10, ("i", "think", "that"): 0.12,
    ("do", "you", "have"): 0.12, ("can", "you", "help"): 0.12, ("do", "you", "want"): 0.12,
    ("how", "are", "you"): 0.15, ("what", "is", "your"): 0.12, ("what", "do", "you"): 0.12,
    ("this", "is", "a"): 0.10, ("that", "is", "a"): 0.10, ("there", "is", "a"): 0.10,
    ("i", "would", "like"): 0.12, ("would", "you", "like"): 0.12, ("would", "like", "to"): 0.10,
    ("in", "the", "world"): 0.08, ("one", "of", "the"): 0.08, ("a", "lot", "of"): 0.08,
    ("as", "well", "as"): 0.08, ("more", "and", "more"): 0.08, ("such", "as", "the"): 0.08,
    ("it", "is", "a"): 0.10, ("it", "was", "a"): 0.10, ("he", "is", "a"): 0.10,
    ("thank", "you", "for"): 0.12, ("nice", "to", "meet"): 0.12, ("good", "to", "see"): 0.10,
}


# =============================================================================
# Curriculum Learning - Shaped Rewards for Word Genome
# =============================================================================

# Subject pronouns that commonly start sentences
SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "who", "what", "this", "that"}

# Verbs for subject-verb detection
COMMON_VERBS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "want", "need", "like", "love", "hate", "think", "know", "see", "hear", "feel",
    "go", "come", "get", "make", "take", "give", "find", "say", "tell", "ask",
    "work", "play", "run", "walk", "eat", "drink", "sleep", "read", "write"
}

# Objects/nouns commonly used as sentence objects
COMMON_OBJECTS = {
    "it", "me", "you", "him", "her", "us", "them", "this", "that",
    "something", "nothing", "everything", "anything",
    "food", "water", "time", "day", "night", "home", "work", "school",
    "help", "money", "love", "life", "world", "way", "thing", "place"
}


def score_structure_curriculum(text: str, phase: int = 1) -> Dict:
    """Score text based on curriculum learning phase with shaped rewards.
    
    Args:
        text: The text to score
        phase: Current curriculum phase (1-4)
            1 = Structure (capitalization, punctuation, spacing)
            2 = Patterns (bigrams, subject-verb)
            3 = Sentences (SVO structure, meaning)
            4 = Conversation (GPT relevance)
    
    Returns:
        Dict with structure_score, individual bonuses, and phase mastery metrics
    """
    if not text or not text.strip():
        return {
            "structure_score": 0.0,
            "has_capital_start": False,
            "has_punctuation_end": False,
            "has_proper_spacing": False,
            "word_count": 0,
            "bigram_count": 0,
            "has_subject_verb": False,
            "has_svo": False,
            "phase_mastery": 0.0
        }
    
    text = text.strip()
    words = [w.strip('.,!?;:').lower() for w in text.split() if w.strip('.,!?;:')]
    
    # === Phase 1: Structure ===
    # Check capitalization
    has_capital_start = text[0].isupper()
    capital_bonus = 0.15 if has_capital_start else 0.0
    
    # Check end punctuation
    has_punctuation_end = text[-1] in '.!?'
    punctuation_bonus = 0.15 if has_punctuation_end else 0.0
    
    # Check proper spacing (2-6 words is ideal)
    word_count = len(words)
    has_proper_spacing = 2 <= word_count <= 6
    spacing_bonus = 0.10 if has_proper_spacing else (0.05 if word_count > 0 else 0.0)
    
    # Phase 1 score
    phase1_score = capital_bonus + punctuation_bonus + spacing_bonus
    phase1_mastery = 1.0 if (has_capital_start and has_punctuation_end) else 0.0
    
    # === Phase 2: Patterns ===
    bigram_count = 0
    has_subject_verb = False
    
    # Detect bigrams
    found_bigrams, found_trigrams = detect_local_patterns(text)
    bigram_count = len(found_bigrams)
    bigram_bonus = min(0.30, bigram_count * 0.15)  # Up to 0.30 for 2+ bigrams
    
    # Check for subject-verb pattern
    for i in range(len(words) - 1):
        if words[i] in SUBJECT_PRONOUNS and words[i + 1] in COMMON_VERBS:
            has_subject_verb = True
            break
    subject_verb_bonus = 0.15 if has_subject_verb else 0.0
    
    # Phase 2 score (includes Phase 1)
    phase2_score = phase1_score + bigram_bonus + subject_verb_bonus
    phase2_mastery = 1.0 if bigram_count >= 1 else 0.0
    
    # === Phase 3: Sentences ===
    has_svo = False
    svo_bonus = 0.0
    
    # Check for Subject-Verb-Object pattern
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if (words[i] in SUBJECT_PRONOUNS and 
                words[i + 1] in COMMON_VERBS and 
                (words[i + 2] in COMMON_OBJECTS or len(words[i + 2]) > 2)):
                has_svo = True
                svo_bonus = 0.30
                break
    
    # Phase 3 score (includes Phase 1 + 2)
    phase3_score = phase2_score + svo_bonus
    phase3_mastery = 1.0 if has_svo else 0.0
    
    # === Total score based on phase ===
    if phase == 1:
        structure_score = phase1_score
        phase_mastery = phase1_mastery
    elif phase == 2:
        structure_score = phase2_score
        phase_mastery = phase2_mastery
    elif phase >= 3:
        structure_score = phase3_score
        phase_mastery = phase3_mastery
    else:
        structure_score = phase1_score
        phase_mastery = phase1_mastery
    
    return {
        "structure_score": min(1.0, structure_score),
        "has_capital_start": has_capital_start,
        "has_punctuation_end": has_punctuation_end,
        "has_proper_spacing": has_proper_spacing,
        "word_count": word_count,
        "bigram_count": bigram_count,
        "trigram_count": len(found_trigrams),
        "has_subject_verb": has_subject_verb,
        "has_svo": has_svo,
        "phase_mastery": phase_mastery,
        "found_bigrams": found_bigrams,
        "found_trigrams": found_trigrams
    }


def detect_local_patterns(text: str) -> Tuple[List[str], List[str]]:
    """Detect common bigrams and trigrams locally without OpenAI.
    
    Returns: (bigrams, trigrams) lists of detected patterns
    """
    words = text.lower().split()
    # Clean punctuation from words
    words = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:')]
    
    found_bigrams = []
    found_trigrams = []
    
    # Check bigrams from predefined list
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if bigram in COMMON_BIGRAMS:
            found_bigrams.append(f"{words[i]} {words[i + 1]}")
    
    # Check trigrams from predefined list
    for i in range(len(words) - 2):
        trigram = (words[i], words[i + 1], words[i + 2])
        if trigram in COMMON_TRIGRAMS:
            found_trigrams.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")
    
    # Also detect grammatical patterns dynamically
    # Subject-verb patterns: pronoun + verb
    pronouns = {"i", "you", "he", "she", "it", "we", "they", "who", "what", "that", "this"}
    verbs = {"am", "is", "are", "was", "were", "have", "has", "had", "do", "does", "did", 
             "will", "would", "can", "could", "should", "may", "might", "must",
             "want", "need", "like", "think", "know", "see", "get", "make", "go", "come"}
    articles = {"a", "an", "the"}
    prepositions = {"in", "on", "at", "to", "for", "with", "by", "from", "of", "about"}
    
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        bigram_str = f"{w1} {w2}"
        if bigram_str in found_bigrams:
            continue
        
        # Pronoun + verb = valid bigram
        if w1 in pronouns and w2 in verbs:
            found_bigrams.append(bigram_str)
        # Article + any word = potentially valid
        elif w1 in articles and len(w2) > 2:
            found_bigrams.append(bigram_str)
        # Preposition + article = valid
        elif w1 in prepositions and w2 in articles:
            found_bigrams.append(bigram_str)
        # Verb + preposition = valid (e.g., "go to", "come from")
        elif w1 in verbs and w2 in prepositions:
            found_bigrams.append(bigram_str)
    
    return found_bigrams, found_trigrams


def detect_flexible_patterns(text: str) -> Dict:
    """Detect patterns using grammatical rules, not just hardcoded lists.
    
    Returns:
        Dict with bigrams, trigrams, and is_valid_sentence
    """
    if not text or not text.strip():
        return {"bigrams": [], "trigrams": [], "is_valid_sentence": False}
    
    text = text.strip()
    words = [w.strip('.,!?;:').lower() for w in text.split() if w.strip('.,!?;:')]
    
    if len(words) < 2:
        return {"bigrams": [], "trigrams": [], "is_valid_sentence": False}
    
    # Word categories for pattern detection
    pronouns = {"i", "you", "he", "she", "it", "we", "they", "who", "what", "that", "this", "these", "those"}
    verbs = {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "done",
        "will", "would", "can", "could", "shall", "should", "may", "might", "must",
        "want", "need", "like", "love", "hate", "think", "know", "see", "hear", "feel",
        "go", "come", "get", "make", "take", "give", "find", "say", "tell", "ask",
        "work", "play", "run", "walk", "eat", "drink", "sleep", "read", "write",
        "look", "seem", "appear", "become", "stay", "keep", "let", "help", "try"
    }
    articles = {"a", "an", "the"}
    prepositions = {"in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into", "onto", "upon"}
    adjectives = {
        "good", "bad", "big", "small", "new", "old", "young", "long", "short", "high", "low",
        "great", "little", "important", "different", "same", "large", "certain", "possible",
        "free", "full", "empty", "open", "closed", "true", "false", "real", "best", "worst",
        "first", "last", "next", "other", "own", "public", "private", "single", "double",
        "happy", "sad", "angry", "quick", "slow", "fast", "easy", "hard", "simple", "complex"
    }
    question_words = {"what", "who", "where", "when", "why", "how", "which", "whose"}
    conjunctions = {"and", "or", "but", "so", "because", "if", "when", "while", "although"}
    
    found_bigrams = []
    found_trigrams = []
    
    # Detect bigrams with grammatical rules
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        bigram_str = f"{w1} {w2}"
        
        # Pronoun + verb (e.g., "I am", "she thinks")
        if w1 in pronouns and w2 in verbs:
            found_bigrams.append(bigram_str)
        # Article + noun/adjective (e.g., "the cat", "a big")
        elif w1 in articles and len(w2) > 2:
            found_bigrams.append(bigram_str)
        # Verb + preposition (e.g., "go to", "look at")
        elif w1 in verbs and w2 in prepositions:
            found_bigrams.append(bigram_str)
        # Preposition + article (e.g., "in the", "on a")
        elif w1 in prepositions and w2 in articles:
            found_bigrams.append(bigram_str)
        # Question word + verb (e.g., "what is", "where do")
        elif w1 in question_words and w2 in verbs:
            found_bigrams.append(bigram_str)
        # Adjective + noun (simple check: adj followed by word > 2 chars)
        elif w1 in adjectives and len(w2) > 2 and w2 not in verbs:
            found_bigrams.append(bigram_str)
        # Also check predefined common bigrams
        elif (w1, w2) in COMMON_BIGRAMS:
            found_bigrams.append(bigram_str)
    
    # Detect trigrams with grammatical rules
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i + 1], words[i + 2]
        trigram_str = f"{w1} {w2} {w3}"
        
        # Subject + verb + object (e.g., "I want something")
        if w1 in pronouns and w2 in verbs and len(w3) > 2:
            found_trigrams.append(trigram_str)
        # Article + adjective + noun (e.g., "the big cat")
        elif w1 in articles and w2 in adjectives and len(w3) > 2:
            found_trigrams.append(trigram_str)
        # Preposition + article + noun (e.g., "in the house")
        elif w1 in prepositions and w2 in articles and len(w3) > 2:
            found_trigrams.append(trigram_str)
        # Question patterns (e.g., "what is the")
        elif w1 in question_words and w2 in verbs:
            found_trigrams.append(trigram_str)
        # Verb + preposition + article (e.g., "go to the")
        elif w1 in verbs and w2 in prepositions and w3 in articles:
            found_trigrams.append(trigram_str)
        # Also check predefined common trigrams
        elif (w1, w2, w3) in COMMON_TRIGRAMS:
            found_trigrams.append(trigram_str)
    
    # Determine if this is a valid sentence
    has_capital = text[0].isupper()
    has_punct = text[-1] in '.!?'
    has_svo = False
    
    # Check for Subject-Verb-Object structure
    for i in range(len(words) - 2):
        if words[i] in pronouns and words[i + 1] in verbs:
            has_svo = True
            break
    
    # Also check for article at start + verb somewhere
    if words[0] in articles:
        if any(w in verbs for w in words[1:]):
            has_svo = True
    
    # Valid sentence: has structure markers AND some grammatical content
    is_valid = has_capital and has_punct and (has_svo or len(found_bigrams) >= 2)
    
    return {
        "bigrams": list(set(found_bigrams)),  # Deduplicate
        "trigrams": list(set(found_trigrams)),
        "is_valid_sentence": is_valid,
        "has_svo": has_svo
    }


def calculate_ngram_bonus(text: str) -> float:
    """Calculate bonus score for common English n-gram patterns.
    
    Returns a bonus (0.0 to 0.5) based on how many common patterns are found.
    """
    if not text:
        return 0.0
    
    words = [w.lower().strip('.,!?') for w in text.split() if w.strip('.,!?')]
    if len(words) < 2:
        return 0.0
    
    bonus = 0.0
    found_patterns = []
    
    # Check bigrams
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if bigram in COMMON_BIGRAMS:
            bonus += COMMON_BIGRAMS[bigram]
            found_patterns.append(bigram)
    
    # Check trigrams
    for i in range(len(words) - 2):
        trigram = (words[i], words[i + 1], words[i + 2])
        if trigram in COMMON_TRIGRAMS:
            bonus += COMMON_TRIGRAMS[trigram]
            found_patterns.append(trigram)
    
    # Cap bonus at 0.5
    return min(0.5, bonus)

# OpenAI client for scoring
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key) if api_key else None


# =============================================================================
# Batch OpenAI Scoring - One API call for all agents
# =============================================================================

async def score_batch_openai(sentences: List[Tuple[str, str]], gpt_context: Optional[str] = None) -> List[Dict]:
    """Score ALL agent sentences in ONE API call for speed and differentiation.
    
    Args:
        sentences: List of (agent_id, text) tuples
        gpt_context: Optional GPT context for relevance scoring
    
    Returns:
        List of score dicts with individual reasoning for each agent
    """
    if not sentences:
        return []
    
    if client is None:
        # No API key - use local scoring fallback
        return [_local_score_fallback(text) for _, text in sentences]
    
    # Build the prompt for batch evaluation
    sentence_list = "\n".join(
        f'{i+1}. "{s[1][:150]}"' for i, s in enumerate(sentences)
    )
    
    context_note = ""
    if gpt_context:
        context_note = f'\nContext (GPT said): "{gpt_context[:100]}"\nAlso rate how well each responds to this context.'
    
    prompt = f"""You are an English teacher evaluating {len(sentences)} sentences from AI agents learning to write.
{context_note}
For EACH sentence, provide scores from 0.0 to 1.0:
- vocab: Are these valid English words? (0.3=some, 0.6=mostly, 1.0=all valid)
- grammar: Is word order grammatical? (0.1=random, 0.3=bad, 0.5=fragments, 0.7=almost, 1.0=correct)
- coherence: Does it make sense? (0.1=nonsense, 0.3=word soup, 0.5=partial, 0.7=understandable, 1.0=clear)

IMPORTANT: Score each sentence INDIVIDUALLY based on its specific quality. Differentiate between them!

Sentences to evaluate:
{sentence_list}

Return ONLY a JSON array with {len(sentences)} objects (one per sentence, in order):
[{{"vocab": 0.X, "grammar": 0.X, "coherence": 0.X, "reason": "brief 5-10 word explanation"}}]"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise English evaluator. Return only valid JSON array. Score each sentence uniquely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        results = json.loads(content)
        
        # Validate and normalize results
        scored_results = []
        for i, (agent_id, text) in enumerate(sentences):
            if i < len(results):
                r = results[i]
                vocab = float(r.get("vocab", 0.5))
                grammar = float(r.get("grammar", 0.3))
                coherence = float(r.get("coherence", 0.3))
                reason = r.get("reason", "")
                
                # Calculate total with weights
                total = vocab * 0.2 + grammar * 0.4 + coherence * 0.4
                
                # Add local pattern detection
                local_patterns = detect_flexible_patterns(text)
                bigrams_found = local_patterns.get("bigrams", [])
                trigrams_found = local_patterns.get("trigrams", [])
                is_valid_sentence = local_patterns.get("is_valid_sentence", False)
                
                # Bonus for patterns found
                pattern_bonus = len(bigrams_found) * 0.02 + len(trigrams_found) * 0.05
                if is_valid_sentence:
                    pattern_bonus += 0.1
                total = min(1.0, total + pattern_bonus)
                
                scored_results.append({
                    "vocabulary_score": vocab,
                    "grammar_score": grammar,
                    "coherence_score": coherence,
                    "total": total,
                    "reason": reason,
                    "discovered_bigrams": bigrams_found,
                    "discovered_trigrams": trigrams_found,
                    "is_valid_sentence": is_valid_sentence,
                    "has_capital_start": text[0].isupper() if text else False,
                    "has_punctuation_end": text[-1] in '.!?' if text else False,
                    "word_count": len(text.split()) if text else 0,
                    "bigram_count": len(bigrams_found),
                    "details": {"reason": reason, "batch_scored": True}
                })
            else:
                # Fallback for missing results
                scored_results.append(_local_score_fallback(text))
        
        print(f"[BATCH SCORING] Scored {len(sentences)} sentences in one call")
        return scored_results
        
    except Exception as e:
        print(f"[BATCH SCORING] Error: {e}, falling back to local scoring")
        return [_local_score_fallback(text) for _, text in sentences]


def _local_score_fallback(text: str) -> Dict:
    """Local scoring fallback when OpenAI is unavailable."""
    if not text or not text.strip():
        return {
            "vocabulary_score": 0.0, "grammar_score": 0.0, "coherence_score": 0.0,
            "total": 0.0, "reason": "Empty", "discovered_bigrams": [],
            "discovered_trigrams": [], "is_valid_sentence": False,
            "has_capital_start": False, "has_punctuation_end": False,
            "word_count": 0, "bigram_count": 0, "details": {}
        }
    
    # Local heuristic scoring
    text = text.strip()
    has_capital = text[0].isupper()
    has_punct = text[-1] in '.!?'
    words = text.split()
    word_count = len(words)
    
    # Vocab check
    vocab_score, _ = quick_vocab_check(text)
    
    # Grammar heuristics
    grammar_score = 0.2
    if has_capital:
        grammar_score += 0.15
    if has_punct:
        grammar_score += 0.15
    if 3 <= word_count <= 12:
        grammar_score += 0.1
    
    # Pattern detection
    patterns = detect_flexible_patterns(text)
    bigrams = patterns.get("bigrams", [])
    trigrams = patterns.get("trigrams", [])
    is_valid = patterns.get("is_valid_sentence", False)
    
    grammar_score += len(bigrams) * 0.05
    grammar_score = min(1.0, grammar_score)
    
    # Coherence based on structure
    coherence_score = 0.3
    if has_capital and has_punct:
        coherence_score += 0.2
    if len(bigrams) > 0:
        coherence_score += 0.15
    if is_valid:
        coherence_score += 0.2
    coherence_score = min(1.0, coherence_score)
    
    total = vocab_score * 0.2 + grammar_score * 0.4 + coherence_score * 0.4
    
    return {
        "vocabulary_score": vocab_score,
        "grammar_score": grammar_score,
        "coherence_score": coherence_score,
        "total": total,
        "reason": f"Local: cap={has_capital}, punct={has_punct}, bigrams={len(bigrams)}",
        "discovered_bigrams": bigrams,
        "discovered_trigrams": trigrams,
        "is_valid_sentence": is_valid,
        "has_capital_start": has_capital,
        "has_punctuation_end": has_punct,
        "word_count": word_count,
        "bigram_count": len(bigrams),
        "details": {"local_scored": True}
    }


# =============================================================================
# Pure OpenAI Scoring with Phase-Specific Rubrics for Word Genome
# =============================================================================

async def score_word_genome_openai(
    sentences: List[Tuple[str, str]],
    phase: int,
    gpt_context: Optional[str] = None
) -> List[Dict]:
    """Score Word Genome agents using GPT-4 with phase-specific rubrics.
    
    This provides accurate, progressive evaluation tailored to the curriculum phase.
    
    Args:
        sentences: List of (agent_id, text) tuples
        phase: Current curriculum phase (1-4)
        gpt_context: GPT's message for Phase 4 relevance scoring
    
    Returns:
        List of score dicts with detailed breakdowns
    """
    if not sentences:
        return []
    
    if client is None:
        print("[WORD GENOME SCORING] No API key - using local fallback")
        return [_local_score_fallback(text) for _, text in sentences]
    
    # Phase-specific rubrics with clear point breakdowns
    rubrics = {
        1: """PHASE 1 - STRUCTURE FOCUS (Total: 100 points)
STRUCTURE (80 points):
- Capital start: First letter is capitalized (25 pts)
- Punctuation end: Ends with . ! or ? (25 pts)
- Word spacing: Proper spaces between all words (15 pts)
- Word count: Between 3-10 words (15 pts)

BASIC QUALITY (20 points):
- Valid words: All/most are real English words (10 pts)
- No repetition: Same word not repeated excessively (10 pts)""",

        2: """PHASE 2 - PATTERN FOCUS (Total: 100 points)
STRUCTURE (60 points):
- Capital start (15 pts)
- Punctuation end (15 pts)
- Proper spacing (15 pts)
- Word count 3-10 (15 pts)

PATTERNS (40 points):
- Valid bigrams: Has article+noun, pronoun+verb, or prep+article patterns (20 pts)
- Common combinations: Uses frequent word pairs (10 pts)
- No word repetition: Each word appears max 2 times (10 pts)""",

        3: """PHASE 3 - SENTENCE FOCUS (Total: 100 points)
STRUCTURE (40 points):
- Capital start (10 pts)
- Punctuation end (10 pts)
- Proper spacing (10 pts)
- Word count 3-10 (10 pts)

PATTERNS (30 points):
- Valid bigrams present (15 pts)
- Common word combinations (15 pts)

COHERENCE (30 points):
- Subject-verb structure: Has identifiable subject and verb (15 pts)
- Semantic sense: Words relate to each other meaningfully (15 pts)""",

        4: """PHASE 4 - CONVERSATION FOCUS (Total: 100 points)
STRUCTURE (20 points):
- Capital start (5 pts)
- Punctuation end (5 pts)
- Proper spacing (5 pts)
- Appropriate length (5 pts)

PATTERNS (20 points):
- Valid bigrams (10 pts)
- Natural word flow (10 pts)

CONVERSATION (60 points):
- Relevance: Response relates to GPT's message (30 pts)
- Response type: Appropriate reply form - answer, question, statement (15 pts)
- Natural dialogue: Sounds like something a person would say (15 pts)"""
    }
    
    # Get the rubric for current phase
    current_rubric = rubrics.get(phase, rubrics[1])
    
    # Build sentence list
    sentence_list = "\n".join(
        f'{i+1}. [Agent {s[0]}]: "{s[1][:200]}"' for i, s in enumerate(sentences)
    )
    
    # Context for Phase 4
    context_section = ""
    if phase == 4 and gpt_context:
        context_section = f'\n\nGPT SAID: "{gpt_context}"\nEvaluate how well each agent RESPONDS to this message.'
    
    prompt = f"""You are evaluating sentences from AI agents learning English through reinforcement learning.

EVALUATION RUBRIC FOR PHASE {phase}:
{current_rubric}
{context_section}

SENTENCES TO EVALUATE:
{sentence_list}

For EACH sentence, calculate the total score (0-100) based on the rubric above.
Be PRECISE and DIFFERENTIATE between sentences - they should NOT all get similar scores.

Return ONLY a JSON array with exactly {len(sentences)} objects in order:
[{{"id": "agent_id", "total": 0-100, "structure": 0-100, "patterns": 0-100, "coherence": 0-100, "conversation": 0-100, "reason": "specific 10-15 word explanation citing rubric criteria"}}]

Scores should reflect the rubric weights:
- Phase 1: structure matters most (80%)
- Phase 2: structure (60%) + patterns (40%)
- Phase 3: structure (40%) + patterns (30%) + coherence (30%)
- Phase 4: structure (20%) + patterns (20%) + conversation (60%)"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise English evaluator for AI language learning. Score each sentence individually using the exact rubric provided. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        results = json.loads(content)
        
        # Process and normalize results
        scored_results = []
        for i, (agent_id, text) in enumerate(sentences):
            if i < len(results):
                r = results[i]
                
                # Get scores (0-100 scale, normalize to 0-1 for consistency)
                total_100 = float(r.get("total", 50))
                structure_100 = float(r.get("structure", 50))
                patterns_100 = float(r.get("patterns", 50))
                coherence_100 = float(r.get("coherence", 50))
                conversation_100 = float(r.get("conversation", 50))
                reason = r.get("reason", "No reason provided")
                
                # Normalize to 0-1 scale
                total = total_100 / 100.0
                structure = structure_100 / 100.0
                patterns = patterns_100 / 100.0
                coherence = coherence_100 / 100.0
                conversation = conversation_100 / 100.0
                
                # Local pattern detection for discoveries
                local_patterns = detect_flexible_patterns(text)
                bigrams_found = local_patterns.get("bigrams", [])
                trigrams_found = local_patterns.get("trigrams", [])
                is_valid_sentence = local_patterns.get("is_valid_sentence", False)
                
                # Structure checks
                has_capital = text[0].isupper() if text else False
                has_punct = text[-1] in '.!?' if text else False
                word_count = len(text.split()) if text else 0
                
                scored_results.append({
                    "vocabulary_score": structure,  # Map structure to vocab for UI
                    "grammar_score": patterns,  # Map patterns to grammar for UI
                    "coherence_score": coherence if phase >= 3 else conversation,
                    "total": total,
                    "reason": reason,
                    "discovered_bigrams": bigrams_found,
                    "discovered_trigrams": trigrams_found,
                    "is_valid_sentence": is_valid_sentence,
                    "has_capital_start": has_capital,
                    "has_punctuation_end": has_punct,
                    "word_count": word_count,
                    "bigram_count": len(bigrams_found),
                    "phase": phase,
                    "details": {
                        "total_100": total_100,
                        "structure_100": structure_100,
                        "patterns_100": patterns_100,
                        "coherence_100": coherence_100,
                        "conversation_100": conversation_100,
                        "reason": reason,
                        "phase_scored": True
                    }
                })
            else:
                # Fallback for missing results
                scored_results.append(_local_score_fallback(text))
        
        print(f"[WORD GENOME SCORING] Phase {phase}: Scored {len(sentences)} sentences with GPT-4")
        return scored_results
        
    except Exception as e:
        print(f"[WORD GENOME SCORING] Error: {e}, falling back to local scoring")
        return [_local_score_fallback(text) for _, text in sentences]


def quick_vocab_check(text: str) -> Tuple[float, List[str]]:
    """Quick vocabulary check using common English words (filters obscure words)."""
    if not text or len(text.strip()) < 2:
        return 0.0, []
    
    words_set = get_common_words()
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    
    if not tokens:
        return 0.0, []
    
    # Accept only common English words (filters obscure/archaic terms)
    valid_words = [t for t in tokens if t in words_set]
    score = len(valid_words) / len(tokens) if tokens else 0.0
    
    # Bonus for proper word separation (natural spacing patterns)
    words_by_space = text.split()
    if len(words_by_space) > 1:
        avg_word_len = sum(len(w) for w in words_by_space) / len(words_by_space)
        if 2 <= avg_word_len <= 8:  # Natural word length range
            score = min(1.0, score + 0.1)  # Small spacing bonus
    
    return score, valid_words


async def score_with_openai(
    text: str, 
    conversation_context: Optional[str] = None,
    is_word_nn: bool = False
) -> Dict:
    """Score text using OpenAI as a holistic judge.
    
    Combines NLTK vocabulary check with OpenAI evaluation.
    Gives partial credit for valid words even if overall text isn't conversational.
    
    Args:
        text: The text to score
        conversation_context: Optional previous conversation context
        is_word_nn: If True, use pure OpenAI scores without heuristic blending
    
    Returns dict with:
        - vocabulary_score: 0-1 (valid English words - primarily from NLTK)
        - grammar_score: 0-1 (grammatical correctness)
        - coherence_score: 0-1 (conversational quality)
        - total: weighted average
        - reason: brief explanation
    """
    if not text or len(text.strip()) < 2:
        return {
            "vocabulary_score": 0.0,
            "grammar_score": 0.0,
            "coherence_score": 0.0,
            "total": 0.0,
            "reason": "Empty or too short"
        }
    
    # Quick check for common vocabulary (fast local check - provides gradient signal)
    vocab_nltk, valid_words = quick_vocab_check(text)
    
    # Grammar heuristics (provides baseline signal)
    has_spaces = ' ' in text
    has_punctuation = any(c in text for c in '.!?')
    starts_upper = text[0].isupper() if text else False
    
    # Basic grammar score from heuristics
    grammar_heuristic = 0.0
    if starts_upper:
        grammar_heuristic += 0.25
    if has_punctuation:
        grammar_heuristic += 0.25
    if has_spaces and len(text.split()) > 1:
        grammar_heuristic += 0.25
    # Check for vowel ratio (readable text)
    vowels = len(re.findall(r'[aeiouAEIOU]', text))
    if len(text) > 0 and 0.15 < vowels / len(text) < 0.6:
        grammar_heuristic += 0.25
    
    if client is None:
        # No API key - use heuristics only
        return {
            "vocabulary_score": vocab_nltk,
            "grammar_score": grammar_heuristic,
            "coherence_score": 0.2 if vocab_nltk > 0.3 else 0.1,
            "total": (vocab_nltk * 0.4 + grammar_heuristic * 0.4 + 0.2 * 0.2),
            "reason": "No API key - using heuristics",
            "valid_words": valid_words
        }
    
    # Only call OpenAI if there's some meaningful content (saves API calls)
    # If vocab_nltk is 0 and no valid patterns, use fast heuristic scoring
    if vocab_nltk == 0 and grammar_heuristic < 0.5:
        return {
            "vocabulary_score": vocab_nltk,
            "grammar_score": grammar_heuristic,
            "coherence_score": 0.0,
            "total": grammar_heuristic * 0.4,  # Only grammar heuristic contributes
            "reason": "No valid words detected - skipping OpenAI call",
            "valid_words": []
        }
    
    # For Word NN outputs (high vocab), we still need OpenAI to evaluate coherence
    # since that's what matters for word sequences
    # Only skip OpenAI if there's truly no content to evaluate
    
    try:
        # Use different prompts for Word NN (known words) vs character-based agents
        if is_word_nn:
            # Word NN always uses valid words, so focus on word ORDER and sentence structure
            # Also identify discovered patterns (bigrams, trigrams) and valid sentences
            prompt = f"""You are an English teacher evaluating word sequences from an AI learning sentence structure.

Text to evaluate: "{text[:200]}"

The AI knows valid English words. Now evaluate how well it ORDERS them into sentences.
Look for these specific patterns and score EACH ONE differently:
- Subject-verb patterns (e.g., "I am", "they have", "he is") = higher grammar
- Common phrases (e.g., "how are you", "what is", "do you") = higher coherence
- Article-noun pairs (e.g., "the dog", "a cat") = higher grammar
- Logical word groupings vs random word soup

IMPORTANT: Score each text UNIQUELY based on its specific word patterns. Vary your scores!

Rate from 0.0-1.0:
1. VOCABULARY: Always 1.0 (all valid words)
2. GRAMMAR: Word order quality (0.1=random, 0.3=some pairs, 0.5=phrase fragments, 0.7=almost sentences, 1.0=proper sentences)
3. COHERENCE: Meaning/intent (0.1=word soup, 0.3=vague topic, 0.5=partial meaning, 0.8=understandable, 1.0=clear message)

ALSO identify any valid English patterns found:
- bigrams: Common two-word pairs like "I am", "you are", "do you", "want to" (list any found)
- trigrams: Common three-word sequences like "how are you", "I want to", "do you have" (list any found)
- is_valid_sentence: true if this forms a coherent, meaningful sentence (not just word fragments)

Return ONLY JSON: {{"vocabulary": 1.0, "grammar": X, "coherence": X, "reason": "specific observation", "bigrams": ["word1 word2"], "trigrams": ["word1 word2 word3"], "is_valid_sentence": true/false}}"""
        elif conversation_context:
            prompt = f"""You are an English teacher evaluating an AI learning to communicate.

Previous context: "{conversation_context[-300:]}"
Student's response: "{text[:200]}"

Be encouraging but accurate. Give partial credit for any progress. Rate:
1. VOCABULARY (0.0-1.0): Any real English words? (0.1=some letters look like words, 0.3=1-2 real words, 0.6=mostly real words, 1.0=all valid vocabulary)
2. GRAMMAR (0.0-1.0): How grammatically correct? (0.2=has word-like patterns, 0.5=some structure, 0.8=minor errors, 1.0=perfect)
3. COHERENCE (0.0-1.0): Relevant response? (0.1=random words, 0.3=topic adjacent, 0.6=reasonable attempt, 1.0=natural response)

Return ONLY JSON: {{"vocabulary": X, "grammar": X, "coherence": X, "reason": "brief"}}"""
        else:
            prompt = f"""You are an English teacher evaluating an AI's first attempt at communication.

Student's text: "{text[:200]}"

Be encouraging but accurate. Give partial credit for any progress. Rate:
1. VOCABULARY (0.0-1.0): Any real English words? (0.1=some letters look like words, 0.3=1-2 real words, 0.6=mostly real words, 1.0=all valid vocabulary)
2. GRAMMAR (0.0-1.0): How grammatically correct? (0.2=has word-like patterns, 0.5=some structure, 0.8=minor errors, 1.0=perfect)
3. COHERENCE (0.0-1.0): Makes sense as a statement? (0.1=random words, 0.3=has meaning, 0.6=understandable, 1.0=natural)

Return ONLY JSON: {{"vocabulary": X, "grammar": X, "coherence": X, "reason": "brief"}}"""
        
        # Log the scoring request for debugging
        print(f"[SCORING] OpenAI request for {'Word NN' if is_word_nn else 'Char'}: {text[:50]}...")
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an English teacher. Score EACH text uniquely based on its specific patterns. Vary scores between texts. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,  # Increased from 0.1 for more varied scoring
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        
        # Log OpenAI's response for debugging
        print(f"[SCORING] OpenAI response: vocab={result.get('vocabulary')}, grammar={result.get('grammar')}, coherence={result.get('coherence')}, reason={result.get('reason', '')[:50]}")
        
        if is_word_nn:
            # For Word NN: Use PURE OpenAI scores without heuristic blending
            # This ensures varied scoring based on word order quality
            vocab_score = 1.0  # Always 1.0 for Word NN (all valid words)
            grammar_score = float(result.get("grammar", 0.2))
            coherence_score = float(result.get("coherence", 0.1))
            
            # Extract discovered patterns from OpenAI
            discovered_bigrams = result.get("bigrams", [])
            discovered_trigrams = result.get("trigrams", [])
            is_valid_sentence = result.get("is_valid_sentence", False)
            
            # Also do LOCAL pattern detection as fallback
            local_bigrams, local_trigrams = detect_local_patterns(text)
            discovered_bigrams = list(set(discovered_bigrams + local_bigrams))
            discovered_trigrams = list(set(discovered_trigrams + local_trigrams))
            
            print(f"[PATTERNS] Found bigrams: {discovered_bigrams}, trigrams: {discovered_trigrams}")
            
            # Word NN uses adjusted weights: emphasize grammar and coherence
            total = (
                vocab_score * 0.1 +  # De-emphasize vocab (always 100%)
                grammar_score * 0.45 +
                coherence_score * 0.45
            )
            
            # Bonus for discovered patterns
            pattern_bonus = len(discovered_bigrams) * 0.05 + len(discovered_trigrams) * 0.08
            if is_valid_sentence:
                pattern_bonus += 0.15
            total = min(1.0, total + pattern_bonus)
        else:
            # For character-based agents: Blend with heuristics
            openai_vocab = float(result.get("vocabulary", 0.3))
            vocab_score = max(vocab_nltk, openai_vocab * 0.7 + vocab_nltk * 0.3)
            
            openai_grammar = float(result.get("grammar", 0.3))
            grammar_score = openai_grammar * 0.7 + grammar_heuristic * 0.3
            
            coherence_score = float(result.get("coherence", 0.2))
            
            # Standard weighted total
            total = (
                vocab_score * config.vocabulary_weight +
                grammar_score * config.grammar_weight +
                coherence_score * config.coherence_weight
            )
        
        # Build response with pattern discoveries for Word NN
        response_dict = {
            "vocabulary_score": vocab_score,
            "grammar_score": grammar_score,
            "coherence_score": coherence_score,
            "total": total,
            "reason": result.get("reason", ""),
            "valid_words": valid_words
        }
        
        # Add pattern discoveries for Word NN
        if is_word_nn:
            response_dict["discovered_bigrams"] = discovered_bigrams
            response_dict["discovered_trigrams"] = discovered_trigrams
            response_dict["is_valid_sentence"] = is_valid_sentence
        
        return response_dict
        
    except Exception as e:
        print(f"OpenAI scoring error: {e}")
        # Fallback to common words vocab + heuristics
        return {
            "vocabulary_score": vocab_nltk,
            "grammar_score": grammar_heuristic,
            "coherence_score": 0.2 if vocab_nltk > 0.3 else 0.1,
            "total": (vocab_nltk * 0.4 + grammar_heuristic * 0.4 + 0.2 * 0.2),
            "reason": f"Error: {str(e)}",
            "valid_words": valid_words
        }


# Keep these for backward compatibility
def score_vocabulary(text: str) -> Tuple[float, Dict]:
    """Score vocabulary using common English word list."""
    score, valid_words = quick_vocab_check(text)
    tokens = re.findall(r'[a-zA-Z]+', text.lower()) if text else []
    return score, {
        "valid_words": len(valid_words),
        "total_words": len(tokens),
        "words_found": valid_words
    }


async def score_grammar(text: str) -> Tuple[float, Dict]:
    """Grammar scoring - now uses OpenAI through unified scoring."""
    # This is kept for backward compatibility but not used directly
    return 0.5, {"method": "openai_unified"}


async def score_coherence(text: str, conversation_context: Optional[str] = None) -> Tuple[float, Dict]:
    """Coherence scoring - now uses OpenAI through unified scoring."""
    # This is kept for backward compatibility but not used directly
    return 0.5, {"method": "openai_unified"}


async def get_openai_response(text: str, conversation_history: List[Dict] = None) -> str:
    """Get OpenAI's conversational response to agent's output.
    
    Args:
        text: Agent's utterance
        conversation_history: List of previous turns [{role, content}]
    
    Returns:
        OpenAI's response string
    """
    if client is None:
        return "I see. Tell me more."
    
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are having a friendly conversation. Respond naturally and briefly (1-2 sentences). Even if the other person's message is unclear, try to engage with it constructively."
            }
        ]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": text[:200]})
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"OpenAI response error: {e}")
        return "That's interesting. Please continue."


async def score_single_output(
    text: str, 
    conversation_context: Optional[str] = None,
    agent_vocabulary: Optional[set] = None,
    is_word_nn: bool = False
) -> Dict:
    """Score a single agent's output using OpenAI as unified judge.
    
    Args:
        text: The text to score
        conversation_context: Optional previous conversation context
        agent_vocabulary: Optional set of words the agent knows
        is_word_nn: If True, use Word NN scoring (pure OpenAI, focus on word order)
    
    Returns dict with:
        - vocabulary_score: 0-1 (valid English words)
        - grammar_score: 0-1 (grammatical correctness)
        - coherence_score: 0-1 (conversational quality)
        - total: weighted combination
        - details: breakdown info
        - vocab_bonus: bonus for using known vocabulary (length-weighted)
    """
    # Use the unified OpenAI scoring
    scores = await score_with_openai(text, conversation_context, is_word_nn=is_word_nn)
    
    total = scores["total"]
    vocab_bonus = 0.0
    ngram_bonus = 0.0
    known_words_used = []
    
    # Calculate n-gram pattern bonus (for Word NN especially)
    if is_word_nn:
        ngram_bonus = calculate_ngram_bonus(text)
        total = min(1.0, total + ngram_bonus)
    
    # Calculate vocabulary bonus weighted by word frequency rank
    if agent_vocabulary and scores["vocabulary_score"] > 0:
        text_words = re.findall(r'\b[a-z]+\b', text.lower())
        known_words_used = [word for word in text_words if word in agent_vocabulary]
        if known_words_used and text_words:
            # Weight bonus by frequency rank:
            # Rank 1-100 = 3x (ultra-common: the, is, you)
            # Rank 101-1000 = 2x (very common: time, good, work)
            # Rank 1001-10000 = 1x (common vocabulary)
            weighted_score = sum(get_word_weight(w) for w in known_words_used)
            vocab_bonus = min(0.3, weighted_score / len(text_words) * 0.1)
            total = min(1.0, total + vocab_bonus)
    
    result = {
        "vocabulary_score": scores["vocabulary_score"],
        "grammar_score": scores["grammar_score"],
        "coherence_score": scores["coherence_score"],
        "total": total,
        "details": {
            "reason": scores.get("reason", ""),
            "valid_words": scores.get("valid_words", []),
            "vocab_bonus": vocab_bonus,
            "ngram_bonus": ngram_bonus,
            "known_words_used": known_words_used
        }
    }
    
    # Pass through discovered patterns for Word NN
    if is_word_nn:
        result["discovered_bigrams"] = scores.get("discovered_bigrams", [])
        result["discovered_trigrams"] = scores.get("discovered_trigrams", [])
        result["is_valid_sentence"] = scores.get("is_valid_sentence", False)
    
    return result


async def score_conversation(
    turns: List[Dict], 
    agent_vocabulary: Optional[set] = None,
    is_word_nn: bool = False
) -> Dict:
    """Score an entire conversation (multiple turns).
    
    Args:
        turns: List of {agent_text, openai_response} dicts
        agent_vocabulary: Optional set of words the agent has discovered
        is_word_nn: If True, use adjusted weights (de-emphasize vocab since it's always 100%)
    
    Returns:
        Aggregated scores across all turns
    """
    if not turns:
        return {
            "vocabulary_score": 0.0,
            "grammar_score": 0.0,
            "coherence_score": 0.0,
            "total": 0.0,
            "details": {"reason": "no turns", "valid_words": []}
        }
    
    all_scores = []
    all_valid_words = []
    context = None
    
    for turn in turns:
        agent_text = turn.get("agent_text", "")
        scores = await score_single_output(agent_text, context, agent_vocabulary, is_word_nn=is_word_nn)
        all_scores.append(scores)
        
        # Collect valid words from each turn
        turn_words = scores.get("details", {}).get("valid_words", [])
        all_valid_words.extend(turn_words)
        
        # Update context for next turn
        openai_response = turn.get("openai_response", "")
        if context:
            context = f"{context} {openai_response}"
        else:
            context = openai_response
    
    # Average scores across turns
    avg_vocab = sum(s["vocabulary_score"] for s in all_scores) / len(all_scores)
    avg_grammar = sum(s["grammar_score"] for s in all_scores) / len(all_scores)
    avg_coherence = sum(s["coherence_score"] for s in all_scores) / len(all_scores)
    
    # Recalculate total with adjusted weights for Word NN
    if is_word_nn:
        # Word NN uses known vocabulary, so vocab is always high
        # Emphasize grammar and coherence instead
        vocab_weight = 0.1
        grammar_weight = 0.45
        coherence_weight = 0.45
    else:
        vocab_weight = config.vocabulary_weight
        grammar_weight = config.grammar_weight
        coherence_weight = config.coherence_weight
    
    avg_total = (
        avg_vocab * vocab_weight +
        avg_grammar * grammar_weight +
        avg_coherence * coherence_weight
    )
    
    # Aggregate vocab bonus from all turns
    total_vocab_bonus = sum(s.get("details", {}).get("vocab_bonus", 0) for s in all_scores)
    all_known_words = []
    for s in all_scores:
        all_known_words.extend(s.get("details", {}).get("known_words_used", []))
    
    # Deduplicate valid words
    unique_valid_words = list(set(all_valid_words))
    
    return {
        "vocabulary_score": avg_vocab,
        "grammar_score": avg_grammar,
        "coherence_score": avg_coherence,
        "total": avg_total,
        "vocab_bonus": total_vocab_bonus,
        "known_words_used": list(set(all_known_words)),
        "turn_scores": all_scores,
        "details": {
            "num_turns": len(turns),
            "best_turn_score": max(s["total"] for s in all_scores),
            "valid_words": unique_valid_words
        }
    }


async def score_all_outputs(outputs: List[Tuple[str, str]]) -> Dict[str, Dict]:
    """Score multiple outputs in parallel.
    
    Args:
        outputs: List of (agent_id, text) tuples
    
    Returns:
        {agent_id: scores_dict}
    """
    tasks = []
    agent_ids = []
    
    for agent_id, text in outputs:
        agent_ids.append(agent_id)
        tasks.append(score_single_output(text))
    
    results = await asyncio.gather(*tasks)
    
    return {
        agent_id: scores 
        for agent_id, scores in zip(agent_ids, results)
    }


# =============================================================================
# Curriculum-Aware Word Genome Scoring (Hybrid Pipeline)
# =============================================================================

async def score_word_genome_output(
    text: str,
    phase: int = 1,
    gpt_context: Optional[str] = None,
    skip_openai: bool = False
) -> Dict:
    """Score Word Genome output using hybrid scoring pipeline.
    
    OPTIMIZED: Phase-specific weights are now handled in score_hybrid().
    Phase 1-2 use fast structure-based scoring (no ML models).
    Phase 3+ use full perplexity + relevance scoring.
    
    Args:
        text: The text to score
        phase: Current curriculum phase (1-4)
        gpt_context: GPT's previous message for relevance scoring
        skip_openai: Ignored (kept for backward compatibility)
    
    Returns:
        Dict with all scoring components
    """
    # Use hybrid scoring - handles all phase-specific weights internally
    hybrid = score_hybrid(text, gpt_context, phase)
    
    # Return with all expected fields for compatibility
    return {
        "total": hybrid["total"],  # Already computed with correct phase weights
        # Hybrid scores
        "perplexity_score": hybrid["perplexity_score"],
        "relevance_score": hybrid["relevance_score"],
        "structure_score": hybrid["structure_score"],
        # Legacy compatibility fields
        "vocabulary_score": hybrid["vocabulary_score"],
        "grammar_score": hybrid["grammar_score"],
        "coherence_score": hybrid["coherence_score"],
        # Structure details
        "has_capital_start": hybrid["has_capital_start"],
        "has_punctuation_end": hybrid["has_punctuation_end"],
        "has_proper_spacing": hybrid["has_proper_spacing"],
        "word_count": hybrid["word_count"],
        "bigram_count": hybrid["bigram_count"],
        "has_subject_verb": hybrid["has_subject_verb"],
        "has_svo": hybrid["has_svo"],
        "phase_mastery": hybrid["phase_mastery"],
        # Pattern discoveries
        "discovered_bigrams": hybrid["discovered_bigrams"],
        "discovered_trigrams": hybrid["discovered_trigrams"],
        "is_valid_sentence": hybrid["is_valid_sentence"],
        # Debug info
        "reason": hybrid.get("details", {}).get("reason", ""),
        "details": hybrid.get("details", {})
    }


def calculate_relevance_bonus(agent_text: str, gpt_text: str) -> float:
    """Calculate bonus for relevance to GPT's message (Phase 4).
    
    Args:
        agent_text: Agent's response
        gpt_text: GPT's previous message
    
    Returns:
        Bonus score (0.0 to 0.30) based on word overlap and response patterns
    """
    if not agent_text or not gpt_text:
        return 0.0
    
    agent_words = set(w.lower().strip('.,!?') for w in agent_text.split())
    gpt_words = set(w.lower().strip('.,!?') for w in gpt_text.split())
    
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "in", "it", "i", "you"}
    agent_content = agent_words - stop_words
    gpt_content = gpt_words - stop_words
    
    if not gpt_content:
        return 0.0
    
    # Calculate word overlap
    overlap = agent_content & gpt_content
    overlap_ratio = len(overlap) / len(gpt_content) if gpt_content else 0.0
    
    # Bonus for word overlap (up to 0.15)
    overlap_bonus = min(0.15, overlap_ratio * 0.30)
    
    # Bonus for question-answer pattern
    qa_bonus = 0.0
    if gpt_text.strip().endswith('?'):
        # GPT asked a question - reward responses that start appropriately
        response_starters = {"yes", "no", "i", "we", "they", "it", "maybe", "sure", "well"}
        first_word = agent_text.split()[0].lower().strip('.,!?') if agent_text.split() else ""
        if first_word in response_starters:
            qa_bonus = 0.10
    
    # Bonus for greeting response
    greeting_bonus = 0.0
    gpt_lower = gpt_text.lower()
    agent_lower = agent_text.lower()
    if any(g in gpt_lower for g in ["hello", "hi", "hey", "how are"]):
        if any(r in agent_lower for r in ["hello", "hi", "hey", "good", "fine", "well"]):
            greeting_bonus = 0.10
    
    return min(0.30, overlap_bonus + qa_bonus + greeting_bonus)
