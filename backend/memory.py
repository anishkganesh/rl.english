"""Pattern memory store for successful n-gram sequences."""

from typing import Dict, List, Tuple
from collections import defaultdict
from config import config


class PatternMemory:
    """Stores successful character patterns and their scores."""
    
    def __init__(self, capacity: int = None):
        self.capacity = capacity or config.memory_capacity
        
        # pattern -> (total_score, count)
        self.patterns: Dict[str, Tuple[float, int]] = {}
        
        # n-gram frequencies from successful outputs
        self.ngram_scores: Dict[int, Dict[str, float]] = defaultdict(dict)
    
    def add_pattern(self, text: str, score: float, min_score_threshold: float = 0.3):
        """Add a successful pattern to memory."""
        if score < min_score_threshold:
            return
        
        # Store the full text
        if text in self.patterns:
            old_score, count = self.patterns[text]
            self.patterns[text] = (old_score + score, count + 1)
        else:
            self.patterns[text] = (score, 1)
        
        # Extract and store n-grams (2 to 5 characters)
        for n in range(2, min(6, len(text) + 1)):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                if ngram in self.ngram_scores[n]:
                    old_score = self.ngram_scores[n][ngram]
                    self.ngram_scores[n][ngram] = (old_score + score) / 2
                else:
                    self.ngram_scores[n][ngram] = score
        
        # Enforce capacity limit
        self._enforce_capacity()
    
    def _enforce_capacity(self):
        """Remove lowest-scoring patterns if over capacity."""
        if len(self.patterns) > self.capacity:
            # Sort by average score
            sorted_patterns = sorted(
                self.patterns.items(),
                key=lambda x: x[1][0] / x[1][1],
                reverse=True
            )
            # Keep only top patterns
            self.patterns = dict(sorted_patterns[:self.capacity])
    
    def get_next_char_bias(self, context: str) -> Dict[str, float]:
        """Get bias for next character based on memory."""
        bias: Dict[str, float] = defaultdict(float)
        
        if not context:
            return dict(bias)
        
        # Look for n-grams that start with our context
        for n in range(2, 6):
            if n <= len(context):
                continue
            
            prefix_len = n - 1
            if len(context) >= prefix_len:
                prefix = context[-prefix_len:]
                
                # Find matching n-grams
                for ngram, score in self.ngram_scores[n].items():
                    if ngram.startswith(prefix):
                        next_char = ngram[-1]
                        bias[next_char] += score * config.memory_bonus
        
        return dict(bias)
    
    def get_top_patterns(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the top N patterns by average score."""
        if not self.patterns:
            return []
        
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1][0] / x[1][1],
            reverse=True
        )
        
        return [(p, s[0]/s[1]) for p, s in sorted_patterns[:n]]
    
    def clear(self):
        """Clear all memory."""
        self.patterns.clear()
        self.ngram_scores.clear()
    
    def to_dict(self) -> dict:
        """Convert memory state to dictionary."""
        return {
            "num_patterns": len(self.patterns),
            "top_patterns": self.get_top_patterns(5)
        }


# Global memory instance
pattern_memory = PatternMemory()

