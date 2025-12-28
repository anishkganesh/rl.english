"""Agent class with genome and action selection logic."""

import random
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from config import config
import uuid


# Common English bigrams - what character often follows after seeing this context
ENGLISH_BIGRAMS = {
    "th": {"e": 0.7, " ": 0.15, "a": 0.1, "i": 0.05},
    "he": {" ": 0.5, "r": 0.2, "l": 0.15, "n": 0.1, "y": 0.05},
    "an": {"d": 0.4, " ": 0.3, "t": 0.15, "y": 0.1, "s": 0.05},
    "in": {"g": 0.4, " ": 0.3, "t": 0.15, "e": 0.1, "s": 0.05},
    "er": {" ": 0.4, "e": 0.2, "s": 0.2, "y": 0.1, "i": 0.1},
    "on": {" ": 0.4, "e": 0.25, "s": 0.15, "t": 0.1, "l": 0.1},
    "re": {" ": 0.3, "a": 0.25, "s": 0.2, "e": 0.15, "d": 0.1},
    "es": {" ": 0.5, "t": 0.25, "s": 0.15, "e": 0.1},
    "or": {" ": 0.4, "e": 0.25, "d": 0.15, "s": 0.1, "t": 0.1},
    "te": {" ": 0.3, "r": 0.25, "d": 0.2, "s": 0.15, "n": 0.1},
    "is": {" ": 0.6, "t": 0.15, "h": 0.1, "s": 0.1, "e": 0.05},
    "it": {" ": 0.5, "h": 0.2, "s": 0.15, "e": 0.1, "y": 0.05},
    "al": {" ": 0.3, "l": 0.25, "s": 0.2, "e": 0.15, "t": 0.1},
    "ar": {" ": 0.3, "e": 0.25, "d": 0.2, "s": 0.15, "t": 0.1},
    "st": {" ": 0.35, "a": 0.2, "e": 0.2, "i": 0.15, "o": 0.1},
    "en": {" ": 0.4, "t": 0.25, "d": 0.15, "s": 0.1, "e": 0.1},
    "nd": {" ": 0.6, "e": 0.15, "s": 0.15, "i": 0.1},
    "to": {" ": 0.5, "o": 0.15, "r": 0.15, "n": 0.1, "p": 0.1},
    "of": {" ": 0.7, "f": 0.15, "t": 0.1, "i": 0.05},
    "ed": {" ": 0.7, "i": 0.1, "e": 0.1, "s": 0.1},
    "ha": {"t": 0.3, "v": 0.25, "s": 0.2, "d": 0.15, " ": 0.1},
    "ng": {" ": 0.6, "s": 0.2, "e": 0.1, "l": 0.1},
    "as": {" ": 0.5, "t": 0.2, "s": 0.15, "e": 0.1, "k": 0.05},
    "at": {" ": 0.4, "e": 0.25, "i": 0.2, "h": 0.1, "t": 0.05},
    " t": {"h": 0.5, "o": 0.2, "a": 0.15, "i": 0.1, "r": 0.05},
    " a": {" ": 0.3, "n": 0.25, "r": 0.15, "l": 0.15, "s": 0.15},
    " i": {" ": 0.3, "n": 0.25, "s": 0.2, "t": 0.15, "f": 0.1},
    " s": {"o": 0.25, "t": 0.25, "a": 0.2, "e": 0.15, "h": 0.15},
    " w": {"a": 0.3, "h": 0.25, "i": 0.2, "e": 0.15, "o": 0.1},
    " h": {"e": 0.4, "a": 0.3, "i": 0.15, "o": 0.1, "u": 0.05},
    " b": {"e": 0.35, "u": 0.25, "y": 0.15, "a": 0.15, "o": 0.1},
    " c": {"o": 0.35, "a": 0.25, "h": 0.2, "u": 0.1, "l": 0.1},
    " m": {"y": 0.25, "a": 0.25, "e": 0.2, "o": 0.15, "i": 0.15},
    " f": {"o": 0.35, "r": 0.25, "i": 0.2, "a": 0.1, "u": 0.1},
    " p": {"r": 0.3, "a": 0.25, "l": 0.2, "e": 0.15, "o": 0.1},
    " d": {"o": 0.35, "i": 0.25, "e": 0.2, "a": 0.1, "u": 0.1},
    " n": {"o": 0.35, "e": 0.25, "i": 0.2, "a": 0.1, "u": 0.1},
    " l": {"i": 0.3, "o": 0.25, "a": 0.2, "e": 0.15, "u": 0.1},
}

# Common English letter frequencies (for default probs)
# Space increased to 18% to encourage natural word separation
LETTER_FREQUENCIES = {
    'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
    'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043,
    'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.024,
    'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019, 'b': 0.015,
    'v': 0.010, 'k': 0.008, 'j': 0.002, 'x': 0.002, 'q': 0.001,
    'z': 0.001, ' ': 0.30, '.': 0.020, ',': 0.015, '!': 0.005, '?': 0.005
}


@dataclass
class AgentGenome:
    """Represents an agent's learned probability distribution over keystrokes."""
    
    # Context -> {key: probability}
    context_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Default probabilities when context not seen
    default_probs: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.default_probs:
            # Initialize with English letter frequencies instead of uniform
            self.default_probs = {}
            for char in config.action_space:
                self.default_probs[char] = LETTER_FREQUENCIES.get(char, 0.01)
            # Normalize
            total = sum(self.default_probs.values())
            self.default_probs = {k: v/total for k, v in self.default_probs.items()}
        
        # Seed with common English bigram patterns
        if not self.context_probs:
            self._seed_with_english_patterns()
    
    def _seed_with_english_patterns(self):
        """Initialize context probabilities with common English patterns."""
        for context, next_chars in ENGLISH_BIGRAMS.items():
            # Create probability distribution for this context
            probs = self.default_probs.copy()
            
            # Boost probabilities for known next characters
            for char, boost in next_chars.items():
                if char in probs:
                    probs[char] = boost
            
            # Normalize
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            self.context_probs[context] = probs
    
    def get_probs(self, context: str) -> Dict[str, float]:
        """Get probability distribution for given context."""
        if context in self.context_probs:
            return self.context_probs[context]
        
        # Check if last 2 chars match any bigram pattern
        if len(context) >= 2:
            last_two = context[-2:]
            if last_two in self.context_probs:
                return self.context_probs[last_two]
        
        return self.default_probs.copy()
    
    def update_context(self, context: str, probs: Dict[str, float]):
        """Update probabilities for a specific context."""
        self.context_probs[context] = probs
    
    def clone(self) -> 'AgentGenome':
        """Create a deep copy of the genome."""
        new_genome = AgentGenome()
        new_genome.default_probs = self.default_probs.copy()
        new_genome.context_probs = {
            ctx: probs.copy() 
            for ctx, probs in self.context_probs.items()
        }
        return new_genome
    
    def to_dict(self) -> dict:
        """Serialize genome to dictionary."""
        return {
            "context_probs": self.context_probs,
            "default_probs": self.default_probs
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentGenome':
        """Deserialize genome from dictionary."""
        genome = cls()
        genome.context_probs = data.get("context_probs", {})
        genome.default_probs = data.get("default_probs", genome.default_probs)
        return genome


class Agent:
    """An agent that types characters based on its genome."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None,
        genome: Optional[AgentGenome] = None,
        exploration_rate: float = None,
        vocabulary: Optional[Set[str]] = None
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.genome = genome or AgentGenome()
        self.exploration_rate = exploration_rate if exploration_rate is not None else config.initial_exploration
        
        # Vocabulary - discovered English words
        self.vocabulary: Set[str] = vocabulary or set()
        
        # Current state
        self.typed_text: str = ""
        self.score: float = 0.0
        self.scores_breakdown: Dict[str, float] = {}
        
    def reset(self, keep_context: bool = False):
        """Reset the agent for a new generation.
        
        Args:
            keep_context: Ignored for genome agents (no context to keep)
        """
        self.typed_text = ""
        self.score = 0.0
        self.scores_breakdown = {}
    
    def add_words(self, words: List[str]):
        """Add discovered words to vocabulary."""
        for word in words:
            if word and len(word) >= 1:
                self.vocabulary.add(word.lower())
    
    def get_context(self) -> str:
        """Get the current context (last N characters)."""
        return self.typed_text[-config.context_length:] if self.typed_text else ""
    
    def select_action(self, memory_bias: Optional[Dict[str, float]] = None) -> str:
        """Select the next character or word to type.
        
        If the agent has vocabulary words, there's a chance it will
        type a whole word instead of a single character.
        """
        context = self.get_context()
        
        # 30% chance to use a vocabulary word if available
        # Only do this at word boundaries (after space or at start)
        at_word_boundary = not self.typed_text or self.typed_text.endswith(' ')
        if self.vocabulary and at_word_boundary and random.random() < 0.3:
            word = random.choice(list(self.vocabulary))
            return word + " "  # Return whole word with trailing space
        
        # Get base probabilities from genome
        probs = self.genome.get_probs(context)
        
        # Apply memory bias if provided
        if memory_bias:
            for char, bonus in memory_bias.items():
                if char in probs:
                    probs[char] = probs[char] * (1 + bonus)
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Random action
            return random.choice(config.action_space)
        else:
            # Sample from distribution
            chars = list(probs.keys())
            weights = [probs[c] for c in chars]
            return random.choices(chars, weights=weights)[0]
    
    def type_character(self, char: str):
        """Add a character to the typed text."""
        self.typed_text += char
    
    def step(self, memory_bias: Optional[Dict[str, float]] = None) -> str:
        """Perform one typing step and return the character typed."""
        char = self.select_action(memory_bias)
        self.type_character(char)
        return char
    
    def set_score(self, score: float, breakdown: Dict[str, float]):
        """Set the agent's fitness score."""
        self.score = score
        self.scores_breakdown = breakdown
    
    def to_dict(self) -> dict:
        """Convert agent state to dictionary for API response."""
        return {
            "id": self.id,
            "typed_text": self.typed_text,
            "score": self.score,
            "scores_breakdown": self.scores_breakdown,
            "exploration_rate": self.exploration_rate,
            "vocabulary": list(self.vocabulary),
            "vocabulary_size": len(self.vocabulary)
        }
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(
            config.min_exploration,
            self.exploration_rate * config.exploration_decay
        )


def create_agent_pool(num_agents: int = None) -> List[Agent]:
    """Create a pool of agents with seeded English pattern genomes.
    
    Each agent starts with the same English pattern base but with slight
    random variations to create diversity.
    """
    n = num_agents or config.num_agents
    agents = []
    
    for _ in range(n):
        agent = Agent()
        
        # Add small random perturbations to create diversity
        for context, probs in agent.genome.context_probs.items():
            for char in probs:
                # Add ±10% noise
                noise = random.uniform(0.9, 1.1)
                probs[char] *= noise
            # Renormalize
            total = sum(probs.values())
            agent.genome.context_probs[context] = {k: v/total for k, v in probs.items()}
        
        agents.append(agent)
    
    return agents

