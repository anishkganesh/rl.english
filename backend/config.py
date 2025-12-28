"""Configuration and hyperparameters for the RL English typing system."""

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # Agent settings
    num_agents: int = 12
    generation_time: float = 3.0  # max seconds per turn (~3s for faster curriculum learning)
    keystroke_interval: float = 0.02  # seconds between keystrokes (faster for curriculum)
    idle_timeout: float = 0.3  # end generation if no output for this long
    
    # Conversation settings
    turns_per_conversation: int = 1  # single turn per generation for faster feedback
    
    # Genome settings
    context_length: int = 4  # characters of context for decisions (increased)
    
    # Evolution settings
    mutation_rate: float = 0.15  # increased for more variation
    mutation_strength: float = 0.3  # how much to perturb probabilities
    elitism_ratio: float = 0.4  # keep top 40% unchanged
    crossover_ratio: float = 0.4  # more breeding from top performers
    
    # Exploration
    initial_exploration: float = 0.5  # start with 50/50 exploration vs learned
    exploration_decay: float = 0.99  # slow decay for longer exploration
    min_exploration: float = 0.02  # allow near-zero randomness eventually
    
    # Action space
    action_space: str = "abcdefghijklmnopqrstuvwxyz .,!?"
    
    # Scoring weights (vocab + grammar = 80%, coherence = 20%)
    vocabulary_weight: float = 0.4
    grammar_weight: float = 0.4
    coherence_weight: float = 0.2
    
    # Memory settings
    memory_capacity: int = 1000  # max patterns to remember
    memory_bonus: float = 0.2  # bonus probability for remembered patterns
    
    # Word NN settings
    word_nn_max_length: int = 15  # max words per generation (no curriculum - always use max)
    
    # Imitation learning
    imitation_rate: float = 0.15  # 15% of training uses GPT patterns
    
    # Word NN conversation settings
    word_nn_turns_per_gen: int = 3  # multi-turn dialogue (3 turns per generation)
    
    # Pattern discovery
    sentence_coherence_threshold: float = 0.5  # min coherence for valid sentence
    max_discovered_sentences: int = 100  # cap on saved sentences
    
    # Chat settings
    max_chat_response_length: int = 20  # max words in chat response


# Global config instance
config = Config()

