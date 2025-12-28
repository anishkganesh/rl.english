"""Word-level genome agent using evolutionary selection on word probabilities."""

import random
import numpy as np
import uuid
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from config import config
from common_words import get_common_words, get_word_ranks


# Get the common words list
SEED_VOCAB = {
    "i", "a", "the", "is", "are", "am", "was", "were", "be",
    "to", "of", "and", "in", "that", "it", "for", "on", "with",
    "he", "she", "they", "we", "you", "my", "your", "his", "her",
    "this", "what", "how", "why", "when", "where", "who",
    "do", "does", "did", "have", "has", "had", "can", "will", "would",
    "not", "no", "yes", "hello", "hi", "good", "bad", "like", "want",
    "think", "know", "see", "go", "come", "get", "make", "take",
    "need", "say", "tell", "ask", "help", "try", "work", "find"
}

def _get_word_list(custom_vocab: Optional[Set[str]] = None) -> List[str]:
    """Get sorted list of words for consistent indexing.
    
    Args:
        custom_vocab: If provided, use this instead of 10k common words.
                     If None, uses 10k common words.
    """
    if custom_vocab is not None:
        # User explicitly chose custom vocab mode
        if len(custom_vocab) > 0:
            print(f"[VOCAB] Using custom vocab with {len(custom_vocab)} words")
            return sorted(list(custom_vocab))
        else:
            # Empty custom vocab - use seed as fallback
            print(f"[VOCAB] Custom vocab empty, using seed vocab ({len(SEED_VOCAB)} words)")
            return sorted(list(SEED_VOCAB))
    # Default: use 10k common words
    words = get_common_words()
    print(f"[VOCAB] Using 10k common words ({len(words)} words)")
    return sorted(list(words))


# Words that commonly start sentences
SENTENCE_STARTERS = {
    # Pronouns (most common starters)
    "i": 0.20, "you": 0.10, "we": 0.08, "they": 0.06, "he": 0.05, "she": 0.05, "it": 0.05,
    # Articles
    "the": 0.08, "a": 0.06, "an": 0.02,
    # Question words
    "what": 0.04, "how": 0.04, "why": 0.03, "when": 0.02, "where": 0.02, "who": 0.02,
    # Common verbs/auxiliaries
    "do": 0.03, "can": 0.02, "will": 0.02, "would": 0.02,
    # Greetings
    "hello": 0.02, "hi": 0.02,
    # Other common starters
    "this": 0.02, "that": 0.02, "my": 0.02, "there": 0.02,
}

# End-of-sentence punctuation
END_PUNCTUATION = {
    ".": 0.70,  # Period most common
    "?": 0.20,  # Questions
    "!": 0.10,  # Exclamations
}


@dataclass
class WordGenome:
    """A genome representing probability distribution over words."""
    
    # Word -> probability mapping
    word_probs: Dict[str, float] = field(default_factory=dict)
    
    # Context-aware probabilities: previous_word -> {word: probability}
    # This allows learning word pairs/sequences
    bigram_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Position-specific probabilities (for curriculum learning)
    start_probs: Dict[str, float] = field(default_factory=dict)  # Words at position 0
    end_tokens: Dict[str, float] = field(default_factory=dict)   # Punctuation at end
    
    # Custom vocabulary (if using discovered words instead of 10k)
    custom_vocab: Optional[Set[str]] = None
    
    def __post_init__(self):
        if not self.word_probs:
            self._initialize_probs()
    
    def _initialize_probs(self):
        """Initialize with frequency-weighted probabilities."""
        words = _get_word_list(self.custom_vocab)
        ranks = get_word_ranks()
        
        # Initialize probabilities based on word frequency rank
        # Lower rank = more common = higher probability
        for word in words:
            rank = ranks.get(word, 5000)
            # Inverse rank weighting: common words get higher probability
            # Use log to prevent extreme skew
            self.word_probs[word] = 1.0 / (np.log(rank + 1) + 1)
        
        # Normalize
        total = sum(self.word_probs.values())
        if total > 0:
            self.word_probs = {w: p / total for w, p in self.word_probs.items()}
        
        # Initialize position-specific probabilities
        self._initialize_start_probs()
        self._initialize_end_tokens()
    
    def _initialize_start_probs(self):
        """Initialize probabilities for sentence-starting words."""
        # Start with sentence starters heavily weighted
        self.start_probs = {}
        
        # Add sentence starters with high probability
        for word, prob in SENTENCE_STARTERS.items():
            if word in self.word_probs or self.custom_vocab is None:
                self.start_probs[word] = prob
        
        # Add remaining words from vocabulary with low probability
        remaining_prob = 1.0 - sum(self.start_probs.values())
        other_words = [w for w in self.word_probs if w not in self.start_probs]
        if other_words and remaining_prob > 0:
            per_word_prob = remaining_prob / len(other_words)
            for word in other_words:
                self.start_probs[word] = per_word_prob
        
        # Normalize
        total = sum(self.start_probs.values())
        if total > 0:
            self.start_probs = {w: p / total for w, p in self.start_probs.items()}
    
    def _initialize_end_tokens(self):
        """Initialize probabilities for end-of-sentence punctuation."""
        self.end_tokens = END_PUNCTUATION.copy()
    
    def get_word_probs(self, prev_word: Optional[str] = None, position: int = -1) -> Dict[str, float]:
        """Get probability distribution over words, optionally conditioned on previous word or position.
        
        Args:
            prev_word: The previous word for bigram context
            position: Word position in sentence (0 = first word, -1 = not specified)
        """
        # Position 0: Use start_probs for sentence-starting words
        if position == 0 and self.start_probs:
            return self.start_probs.copy()
        
        # Bigram context takes priority after position 0
        if prev_word and prev_word in self.bigram_probs:
            return self.bigram_probs[prev_word]
        
        return self.word_probs.copy()
    
    def sample_word(self, prev_word: Optional[str] = None, temperature: float = 1.0, position: int = -1) -> str:
        """Sample a word from the distribution.
        
        Args:
            prev_word: Previous word for context
            temperature: Sampling temperature (higher = more random)
            position: Word position (0 = first word)
        """
        probs = self.get_word_probs(prev_word, position)
        
        if not probs:
            return "hello"  # Fallback
        
        if temperature != 1.0:
            # Apply temperature scaling
            words = list(probs.keys())
            weights = np.array([probs[w] for w in words])
            weights = weights ** (1.0 / temperature)
            weights = weights / weights.sum()
            return np.random.choice(words, p=weights)
        else:
            words = list(probs.keys())
            weights = [probs[w] for w in words]
            return random.choices(words, weights=weights)[0]
    
    def sample_end_punctuation(self, temperature: float = 1.0) -> str:
        """Sample end-of-sentence punctuation."""
        if not self.end_tokens:
            return "."  # Default to period
        
        tokens = list(self.end_tokens.keys())
        weights = [self.end_tokens[t] for t in tokens]
        
        if temperature != 1.0:
            weights = np.array(weights) ** (1.0 / temperature)
            weights = weights / weights.sum()
        
        return random.choices(tokens, weights=weights)[0]
    
    def clone(self) -> 'WordGenome':
        """Create a deep copy of the genome."""
        new_genome = WordGenome.__new__(WordGenome)
        new_genome.word_probs = self.word_probs.copy()
        new_genome.bigram_probs = {
            k: v.copy() for k, v in self.bigram_probs.items()
        }
        new_genome.start_probs = self.start_probs.copy()
        new_genome.end_tokens = self.end_tokens.copy()
        new_genome.custom_vocab = self.custom_vocab
        return new_genome
    
    def to_dict(self) -> dict:
        """Serialize genome to dictionary."""
        # Only save top probabilities to reduce size
        sorted_probs = sorted(self.word_probs.items(), key=lambda x: x[1], reverse=True)
        top_probs = dict(sorted_probs[:500])  # Keep top 500
        
        # Save top start_probs
        sorted_start = sorted(self.start_probs.items(), key=lambda x: x[1], reverse=True)
        top_start = dict(sorted_start[:100])  # Keep top 100 starters
        
        return {
            "word_probs": top_probs,
            "bigram_probs": self.bigram_probs,
            "start_probs": top_start,
            "end_tokens": self.end_tokens
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WordGenome':
        """Deserialize genome from dictionary."""
        genome = cls()
        if "word_probs" in data:
            # Merge saved probs with default
            for word, prob in data["word_probs"].items():
                if word in genome.word_probs:
                    genome.word_probs[word] = prob
            # Renormalize
            total = sum(genome.word_probs.values())
            genome.word_probs = {w: p / total for w, p in genome.word_probs.items()}
        if "bigram_probs" in data:
            genome.bigram_probs = data["bigram_probs"]
        if "start_probs" in data:
            # Merge saved start_probs
            for word, prob in data["start_probs"].items():
                if word in genome.start_probs:
                    genome.start_probs[word] = prob
            # Renormalize
            total = sum(genome.start_probs.values())
            if total > 0:
                genome.start_probs = {w: p / total for w, p in genome.start_probs.items()}
        if "end_tokens" in data:
            genome.end_tokens = data["end_tokens"]
        return genome


class WordGenomeAgent:
    """An agent that generates sentences using word-level genome probabilities."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        genome: Optional[WordGenome] = None,
        exploration_rate: float = None,
        custom_vocab: Optional[Set[str]] = None
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.custom_vocab = custom_vocab  # Store for later use
        self.genome = genome or WordGenome(custom_vocab=custom_vocab)
        self.exploration_rate = exploration_rate if exploration_rate is not None else config.initial_exploration
        
        # Current state
        self.typed_text: str = ""
        self.words_generated: List[str] = []
        self.score: float = 0.0
        self.scores_breakdown: Dict[str, float] = {}
        
        # Temperature for sampling
        self.temperature = 1.0
        
        # Conversation context
        self.context_words: List[str] = []
        
        # Curriculum phase for adaptive sentence length
        self.current_phase: int = 1
        
        # Lineage tracking
        self.origin: str = "random"  # "elite", "crossover", "mutation", "random"
        self.parent_ids: List[str] = []  # Parent agent IDs
        self.generation_born: int = 0  # Which generation this agent was created
        self.lineage: List[Dict] = []  # Full ancestry tree [{gen, id, origin, parent_ids}]
    
    def reset(self, keep_context: bool = False):
        """Reset for new generation."""
        self.typed_text = ""
        self.words_generated = []
        self.score = 0.0
        self.scores_breakdown = {}
        # Reset target length for new sentence
        if hasattr(self, '_current_target_length'):
            delattr(self, '_current_target_length')
        if not keep_context:
            self.context_words = []
    
    def set_phase(self, phase: int):
        """Set the curriculum learning phase for adaptive sentence length.
        
        Args:
            phase: 1-4
                1-2: Structure/Patterns (4-6 words)
                3: Sentences (6-10 words)
                4: Conversation (10-15 words)
        """
        self.current_phase = phase
    
    def _get_length_params(self) -> tuple:
        """Get variable sentence length parameters with proper distribution.
        
        Target distribution for natural sentence variety:
        - 3 words: ~15%
        - 4 words: ~20%
        - 5 words: ~25%
        - 6 words: ~20%
        - 7 words: ~12%
        - 8-10 words: ~8%
        
        Returns:
            Tuple of (min_words, max_words, stop_check_start, stop_prob_start, stop_prob_increment)
        """
        min_words = 3
        max_words = 10
        
        # Start checking stop probability at 3 words (not at a random target)
        # This ensures we get short sentences (3-4 words) with reasonable frequency
        stop_check_start = 3
        
        # Aggressive stop probability curve:
        # At 3 words: 15% chance to stop
        # At 4 words: 35% chance to stop  
        # At 5 words: 55% chance to stop
        # At 6 words: 75% chance to stop
        # At 7+ words: 90%+ chance to stop
        stop_prob_start = 0.15
        stop_increment = 0.20
        
        return (min_words, max_words, stop_check_start, stop_prob_start, stop_increment)
    
    def step(self) -> str:
        """Generate the next word with variable sentence length."""
        # Get length parameters
        min_words, max_words, stop_check_start, stop_prob_start, stop_increment = self._get_length_params()
        
        num_words = len(self.words_generated)
        
        # Hard limit on word count
        if num_words >= max_words:
            return ""
        
        # Variable-length sentences: start checking stop at 3 words
        # This creates natural distribution: more short sentences, fewer long ones
        if num_words >= stop_check_start:
            # Calculate stop probability based on how many words over the minimum
            words_over = num_words - stop_check_start
            stop_prob = min(0.95, stop_prob_start + words_over * stop_increment)
            
            if random.random() < stop_prob:
                return ""  # Stop generating - sentence is complete
        
        # Get previous word for context
        prev_word = self.words_generated[-1] if self.words_generated else None
        position = len(self.words_generated)  # 0 for first word
        
        # Exploration: random word from vocabulary
        if random.random() < self.exploration_rate:
            words = _get_word_list(self.custom_vocab)
            word = random.choice(words) if words else "hello"
        else:
            # Exploitation: sample from genome distribution with position awareness
            word = self.genome.sample_word(prev_word, self.temperature, position)
        
        # Add word to output
        self.words_generated.append(word)
        
        # Capitalize first word
        display_word = word.capitalize() if position == 0 else word
        
        if self.typed_text:
            self.typed_text += " " + display_word
        else:
            self.typed_text = display_word
        
        return display_word + " "
    
    def finalize_sentence(self) -> str:
        """Add punctuation to the end of the sentence."""
        if not self.typed_text:
            return ""
        
        # Sample end punctuation from genome
        punct = self.genome.sample_end_punctuation(self.temperature)
        
        # Remove trailing space and add punctuation
        self.typed_text = self.typed_text.rstrip() + punct
        
        return punct
    
    def generate_sentence(self, max_words: int = 15) -> str:
        """Generate a complete sentence."""
        self.reset(keep_context=True)
        
        for _ in range(max_words):
            word = self.step()
            if not word:
                break
        
        return self.typed_text
    
    def set_score(self, score: float, breakdown: Dict[str, float]):
        """Set the agent's fitness score."""
        self.score = score
        self.scores_breakdown = breakdown
    
    def set_context(self, gpt_response: str):
        """Set conversation context from GPT response.
        
        Extracts words for context-aware generation in next turn.
        """
        if not gpt_response:
            return
        # Extract words from GPT response
        words = gpt_response.lower().split()
        self.context_words = [w.strip('.,!?') for w in words if w.strip('.,!?')]
    
    def train_on_sequence(self, words: List[str]):
        """No-op for genome agents (no gradient-based learning).
        
        Genome agents learn through evolutionary selection, not imitation.
        """
        pass  # Genome agents don't do gradient-based imitation learning
    
    @property
    def vocabulary(self) -> Set[str]:
        """Get vocabulary as set (for UI compatibility)."""
        # Return top words by probability
        sorted_words = sorted(self.genome.word_probs.items(), key=lambda x: x[1], reverse=True)
        return set(w for w, _ in sorted_words[:100])
    
    def to_dict(self) -> dict:
        """Convert agent state to dictionary for API response."""
        # Extract top 15 words by probability for card display
        sorted_words = sorted(
            self.genome.word_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:15]
        top_words = [{"word": w, "prob": p} for w, p in sorted_words]
        
        return {
            "id": self.id,
            "typed_text": self.typed_text,
            "score": self.score,
            "scores_breakdown": self.scores_breakdown,
            "exploration_rate": self.exploration_rate,
            "vocabulary": list(self.vocabulary),
            "vocabulary_size": len(self.vocabulary),
            "model_type": "word_genome",
            # Lineage tracking
            "origin": self.origin,
            "parent_ids": self.parent_ids,
            "generation_born": self.generation_born,
            "lineage": self.lineage[:10],  # Limit depth for API response
            # Word probabilities
            "top_words": top_words
        }
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(
            config.min_exploration,
            self.exploration_rate * config.exploration_decay
        )
    
    def add_words(self, words: Set[str]):
        """Add words to vocabulary - no-op for Word Genome.
        
        Word Genome already uses the full 10k common words list.
        """
        pass  # Word Genome already has all common words
    
    def update_vocabulary(self, global_vocab: Set[str]):
        """Update vocabulary - no-op for Word Genome.
        
        Word Genome uses its own probability-weighted vocabulary.
        """
        pass
    
    def get_visualization_data(self) -> dict:
        """Get visualization data for UI.
        
        Returns simplified data for Word Genome (no neural network).
        """
        # Return top word probabilities as "weights"
        sorted_words = sorted(
            self.genome.word_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        return {
            "type": "word_genome",
            "top_words": [{"word": w, "prob": p} for w, p in sorted_words],
            "exploration_rate": self.exploration_rate,
            "total_words": len(self.genome.word_probs)
        }
    
    def save(self, path: str):
        """Save agent genome to file."""
        import json
        from pathlib import Path
        
        save_path = Path(path) / f"word_genome_{self.id}.json"
        
        # Only save top probabilities to reduce file size
        sorted_word_probs = sorted(self.genome.word_probs.items(), key=lambda x: x[1], reverse=True)
        sorted_start_probs = sorted(self.genome.start_probs.items(), key=lambda x: x[1], reverse=True)
        
        data = {
            "id": self.id,
            "exploration_rate": self.exploration_rate,
            "word_probs": dict(sorted_word_probs[:500]),  # Top 500
            "bigram_probs": self.genome.bigram_probs,
            "start_probs": dict(sorted_start_probs[:100]),  # Top 100 starters
            "end_tokens": self.genome.end_tokens
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> Optional['WordGenomeAgent']:
        """Load agent genome from file."""
        import json
        from pathlib import Path
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            agent = cls(agent_id=data['id'])
            agent.exploration_rate = data.get('exploration_rate', 0.5)
            
            # Load word probabilities
            if 'word_probs' in data:
                for word, prob in data['word_probs'].items():
                    if word in agent.genome.word_probs:
                        agent.genome.word_probs[word] = prob
                # Renormalize
                total = sum(agent.genome.word_probs.values())
                if total > 0:
                    agent.genome.word_probs = {w: p / total for w, p in agent.genome.word_probs.items()}
            
            # Load bigram probabilities
            agent.genome.bigram_probs = data.get('bigram_probs', {})
            
            # Load position-aware probabilities
            if 'start_probs' in data:
                for word, prob in data['start_probs'].items():
                    if word in agent.genome.start_probs:
                        agent.genome.start_probs[word] = prob
                # Renormalize
                total = sum(agent.genome.start_probs.values())
                if total > 0:
                    agent.genome.start_probs = {w: p / total for w, p in agent.genome.start_probs.items()}
            
            if 'end_tokens' in data:
                agent.genome.end_tokens = data['end_tokens']
            
            return agent
        except Exception as e:
            print(f"[WARN] Failed to load agent from {path}: {e}")
            return None


# =============================================================================
# Evolution Functions for Word Genome
# =============================================================================

def mutate_word_genome(genome: WordGenome, mutation_rate: float = None, strength: float = None) -> WordGenome:
    """Create a mutated copy of a word genome."""
    rate = mutation_rate if mutation_rate is not None else config.mutation_rate
    strength = strength if strength is not None else config.mutation_strength
    
    new_genome = genome.clone()
    
    # Mutate word probabilities
    for word in new_genome.word_probs:
        if random.random() < rate:
            # Add gaussian noise
            delta = random.gauss(0, strength)
            new_genome.word_probs[word] = max(0.0001, new_genome.word_probs[word] + delta)
    
    # Normalize
    total = sum(new_genome.word_probs.values())
    new_genome.word_probs = {w: p / total for w, p in new_genome.word_probs.items()}
    
    # Mutate start_probs (position-aware)
    for word in new_genome.start_probs:
        if random.random() < rate:
            delta = random.gauss(0, strength)
            new_genome.start_probs[word] = max(0.0001, new_genome.start_probs[word] + delta)
    
    # Normalize start_probs
    total = sum(new_genome.start_probs.values())
    if total > 0:
        new_genome.start_probs = {w: p / total for w, p in new_genome.start_probs.items()}
    
    # Mutate end_tokens
    for token in new_genome.end_tokens:
        if random.random() < rate:
            delta = random.gauss(0, strength * 0.5)  # Smaller mutation for punctuation
            new_genome.end_tokens[token] = max(0.01, new_genome.end_tokens[token] + delta)
    
    # Normalize end_tokens
    total = sum(new_genome.end_tokens.values())
    if total > 0:
        new_genome.end_tokens = {t: p / total for t, p in new_genome.end_tokens.items()}
    
    # Mutate bigram probabilities
    for prev_word in new_genome.bigram_probs:
        for word in new_genome.bigram_probs[prev_word]:
            if random.random() < rate:
                delta = random.gauss(0, strength)
                new_genome.bigram_probs[prev_word][word] = max(
                    0.0001,
                    new_genome.bigram_probs[prev_word][word] + delta
                )
        # Normalize
        total = sum(new_genome.bigram_probs[prev_word].values())
        if total > 0:
            new_genome.bigram_probs[prev_word] = {
                w: p / total for w, p in new_genome.bigram_probs[prev_word].items()
            }
    
    return new_genome


def crossover_word_genomes(genome1: WordGenome, genome2: WordGenome) -> WordGenome:
    """Create a child genome by combining two parent word genomes."""
    child = WordGenome()
    
    # Crossover word probabilities (weighted average with random blend)
    all_words = set(genome1.word_probs.keys()) | set(genome2.word_probs.keys())
    
    for word in all_words:
        p1 = genome1.word_probs.get(word, 0.0001)
        p2 = genome2.word_probs.get(word, 0.0001)
        
        # Random blend
        blend = random.random()
        child.word_probs[word] = blend * p1 + (1 - blend) * p2
    
    # Normalize
    total = sum(child.word_probs.values())
    child.word_probs = {w: p / total for w, p in child.word_probs.items()}
    
    # Crossover start_probs (position-aware)
    all_start_words = set(genome1.start_probs.keys()) | set(genome2.start_probs.keys())
    for word in all_start_words:
        p1 = genome1.start_probs.get(word, 0.0001)
        p2 = genome2.start_probs.get(word, 0.0001)
        blend = random.random()
        child.start_probs[word] = blend * p1 + (1 - blend) * p2
    
    # Normalize start_probs
    total = sum(child.start_probs.values())
    if total > 0:
        child.start_probs = {w: p / total for w, p in child.start_probs.items()}
    
    # Crossover end_tokens
    all_tokens = set(genome1.end_tokens.keys()) | set(genome2.end_tokens.keys())
    for token in all_tokens:
        p1 = genome1.end_tokens.get(token, 0.01)
        p2 = genome2.end_tokens.get(token, 0.01)
        blend = random.random()
        child.end_tokens[token] = blend * p1 + (1 - blend) * p2
    
    # Normalize end_tokens
    total = sum(child.end_tokens.values())
    if total > 0:
        child.end_tokens = {t: p / total for t, p in child.end_tokens.items()}
    
    # Crossover bigram probabilities
    all_contexts = set(genome1.bigram_probs.keys()) | set(genome2.bigram_probs.keys())
    
    for context in all_contexts:
        child.bigram_probs[context] = {}
        
        g1_context = genome1.bigram_probs.get(context, {})
        g2_context = genome2.bigram_probs.get(context, {})
        all_next_words = set(g1_context.keys()) | set(g2_context.keys())
        
        for word in all_next_words:
            p1 = g1_context.get(word, 0.0001)
            p2 = g2_context.get(word, 0.0001)
            blend = random.random()
            child.bigram_probs[context][word] = blend * p1 + (1 - blend) * p2
        
        # Normalize
        total = sum(child.bigram_probs[context].values())
        if total > 0:
            child.bigram_probs[context] = {
                w: p / total for w, p in child.bigram_probs[context].items()
            }
    
    return child


def learn_from_success_word_genome(agent: WordGenomeAgent, structure_score: float = 0.0):
    """Update an agent's word genome based on successful output.
    
    Args:
        agent: The agent to update
        structure_score: Extra boost if agent achieved structure goals (capitalization, punctuation)
    """
    if agent.score < 0.15:  # Lower threshold for early learning
        return
    
    words = agent.words_generated
    if len(words) < 1:
        return
    
    # Base boost proportional to score
    boost = agent.score * 0.05
    
    # Extra boost for structure achievement
    if structure_score > 0:
        boost += structure_score * 0.03
    
    # Learn start word pattern (position 0)
    if words:
        first_word = words[0].lower()
        if first_word in agent.genome.start_probs:
            agent.genome.start_probs[first_word] = min(
                0.5,
                agent.genome.start_probs[first_word] + boost * 1.5  # Extra weight for start words
            )
            # Normalize start_probs
            total = sum(agent.genome.start_probs.values())
            if total > 0:
                agent.genome.start_probs = {w: p / total for w, p in agent.genome.start_probs.items()}
    
    # Learn punctuation preference based on success
    # Check what punctuation was used
    text = agent.typed_text
    if text:
        last_char = text[-1]
        if last_char in agent.genome.end_tokens:
            agent.genome.end_tokens[last_char] = min(
                0.8,
                agent.genome.end_tokens[last_char] + boost
            )
            # Normalize end_tokens
            total = sum(agent.genome.end_tokens.values())
            if total > 0:
                agent.genome.end_tokens = {t: p / total for t, p in agent.genome.end_tokens.items()}
    
    # Boost probability of words that appeared in successful output
    for word in words:
        word_lower = word.lower()
        if word_lower in agent.genome.word_probs:
            agent.genome.word_probs[word_lower] = min(0.5, agent.genome.word_probs[word_lower] + boost)
    
    # Learn bigram patterns (for position > 0)
    for i in range(1, len(words)):
        prev_word = words[i - 1].lower()
        curr_word = words[i].lower()
        
        if prev_word not in agent.genome.bigram_probs:
            # Initialize from default probs
            agent.genome.bigram_probs[prev_word] = agent.genome.word_probs.copy()
        
        # Boost the observed sequence
        agent.genome.bigram_probs[prev_word][curr_word] = min(
            0.5,
            agent.genome.bigram_probs[prev_word].get(curr_word, 0.0001) + boost
        )
        
        # Normalize
        total = sum(agent.genome.bigram_probs[prev_word].values())
        agent.genome.bigram_probs[prev_word] = {
            w: p / total for w, p in agent.genome.bigram_probs[prev_word].items()
        }
    
    # Normalize word probs
    total = sum(agent.genome.word_probs.values())
    agent.genome.word_probs = {w: p / total for w, p in agent.genome.word_probs.items()}


def evolve_word_genome_generation(agents: List[WordGenomeAgent], current_generation: int = 0) -> List[WordGenomeAgent]:
    """Create the next generation of word genome agents through evolution.
    
    Args:
        agents: Current generation's agents
        current_generation: Current generation number (for lineage tracking)
    """
    num_agents = len(agents)
    next_gen = current_generation + 1
    
    # Sort by score
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    
    # Learn from successful agents
    for agent in sorted_agents[:3]:
        learn_from_success_word_genome(agent)
    
    def build_lineage(agent: WordGenomeAgent) -> List[Dict]:
        """Build lineage tree from an agent."""
        entry = {
            "gen": agent.generation_born,
            "id": agent.id,
            "origin": agent.origin,
            "parent_ids": agent.parent_ids.copy()
        }
        return [entry] + agent.lineage.copy()
    
    new_agents: List[WordGenomeAgent] = []
    
    # Elitism: Keep top performers unchanged
    num_elite = max(1, int(num_agents * config.elitism_ratio))
    for i in range(num_elite):
        parent = sorted_agents[i]
        elite = WordGenomeAgent(
            genome=parent.genome.clone(),
            exploration_rate=parent.exploration_rate
        )
        # Set lineage for elite
        elite.origin = "elite"
        elite.parent_ids = [parent.id]
        elite.generation_born = next_gen
        elite.lineage = build_lineage(parent)
        elite.decay_exploration()
        new_agents.append(elite)
    
    # Crossover: Create some through breeding
    num_crossover = int(num_agents * config.crossover_ratio)
    parents = sorted_agents[:max(2, num_elite)]
    
    for _ in range(num_crossover):
        p1, p2 = random.sample(parents, 2)
        child_genome = crossover_word_genomes(p1.genome, p2.genome)
        child_genome = mutate_word_genome(child_genome, mutation_rate=config.mutation_rate / 2)
        
        child = WordGenomeAgent(
            genome=child_genome,
            exploration_rate=(p1.exploration_rate + p2.exploration_rate) / 2
        )
        # Set lineage for crossover child
        child.origin = "crossover"
        child.parent_ids = [p1.id, p2.id]
        child.generation_born = next_gen
        # Combine lineages from both parents
        child.lineage = build_lineage(p1) + build_lineage(p2)
        child.decay_exploration()
        new_agents.append(child)
    
    # Mutation: Fill the rest with mutated copies of top performers
    while len(new_agents) < num_agents:
        parent = random.choice(parents)
        mutated_genome = mutate_word_genome(parent.genome)
        
        child = WordGenomeAgent(
            genome=mutated_genome,
            exploration_rate=parent.exploration_rate
        )
        # Set lineage for mutation
        child.origin = "mutation"
        child.parent_ids = [parent.id]
        child.generation_born = next_gen
        child.lineage = build_lineage(parent)
        child.decay_exploration()
        new_agents.append(child)
    
    return new_agents


def create_word_genome_agent_pool(
    num_agents: int = None, 
    load_existing: bool = True,
    custom_vocab: Optional[Set[str]] = None
) -> List[WordGenomeAgent]:
    """Create a pool of word genome agents with slight random variations.
    
    Args:
        num_agents: Number of agents to create.
        load_existing: Whether to load existing saved agents.
        custom_vocab: If provided, use this instead of 10k common words.
    """
    from pathlib import Path
    import json
    
    n = num_agents or config.num_agents
    agents = []
    
    # Try to load existing agents (only if not using custom vocab)
    save_dir = Path("saved_agents/word_genome")
    if load_existing and save_dir.exists() and custom_vocab is None:
        for path in save_dir.glob("*.json"):
            try:
                agent = WordGenomeAgent.load(str(path))
                if agent:
                    agents.append(agent)
                    if len(agents) >= n:
                        break
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
    
    # Fill remaining with new agents
    while len(agents) < n:
        agent = WordGenomeAgent(custom_vocab=custom_vocab)
        
        # Add small random perturbations to create diversity
        for word in list(agent.genome.word_probs.keys()):
            # Add ±20% noise
            noise = random.uniform(0.8, 1.2)
            agent.genome.word_probs[word] *= noise
        
        # Normalize
        total = sum(agent.genome.word_probs.values())
        agent.genome.word_probs = {w: p / total for w, p in agent.genome.word_probs.items()}
        
        agents.append(agent)
    
    return agents


def save_word_genome_agents(agents: List[WordGenomeAgent]):
    """Save all word genome agents to disk."""
    from pathlib import Path
    
    save_dir = Path("saved_agents/word_genome")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save top agents by score
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    for agent in sorted_agents[:5]:  # Save top 5
        agent.save(str(save_dir))

