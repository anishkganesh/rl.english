"""Word-level transformer agent for sentence generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
import uuid
import math


# Small seed vocabulary to get started (common function words)
SEED_VOCABULARY = {
    "i", "a", "the", "is", "are", "am", "was", "were", "be",
    "to", "of", "and", "in", "that", "it", "for", "on", "with",
    "he", "she", "they", "we", "you", "my", "your", "his", "her",
    "this", "what", "how", "why", "when", "where", "who",
    "do", "does", "did", "have", "has", "had", "can", "will", "would",
    "not", "no", "yes", "hello", "hi", "good", "bad", "like", "want"
}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]

# Model hyperparameters
MAX_VOCAB_SIZE = 500  # Max vocabulary size
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
MAX_SEQ_LENGTH = 16  # Max words in sentence


class WordVocabulary:
    """Manages word-to-index mapping."""
    
    def __init__(self, words: Optional[Set[str]] = None, use_10k: bool = False):
        """Initialize vocabulary.
        
        Args:
            words: Custom words to add (typically discovered vocabulary)
            use_10k: If True, use 10k common words. If False, use seed + custom words.
        """
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        
        # Add special tokens
        for i, token in enumerate(SPECIAL_TOKENS):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        # Add seed vocabulary
        for word in SEED_VOCABULARY:
            self.add_word(word)
        
        # Use 10k common words if requested
        if use_10k:
            from common_words import get_common_words
            for word in get_common_words():
                self.add_word(word)
        # Otherwise add provided custom words
        elif words:
            for word in words:
                self.add_word(word)
    
    def add_word(self, word: str) -> int:
        """Add a word to vocabulary, return its index."""
        word = word.lower().strip()
        if not word or word in self.word_to_idx:
            return self.word_to_idx.get(word, self.word_to_idx[UNK_TOKEN])
        
        if len(self.word_to_idx) >= MAX_VOCAB_SIZE:
            return self.word_to_idx[UNK_TOKEN]
        
        idx = len(self.word_to_idx)
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        return idx
    
    def get_idx(self, word: str) -> int:
        """Get index for a word."""
        return self.word_to_idx.get(word.lower(), self.word_to_idx[UNK_TOKEN])
    
    def get_word(self, idx: int) -> str:
        """Get word for an index."""
        return self.idx_to_word.get(idx, UNK_TOKEN)
    
    def __len__(self) -> int:
        return len(self.word_to_idx)
    
    def get_all_words(self) -> List[str]:
        """Get all words (excluding special tokens)."""
        return [w for w in self.word_to_idx.keys() if w not in SPECIAL_TOKENS]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class WordTransformer(nn.Module):
    """Transformer for word-level text generation."""
    
    def __init__(
        self,
        vocab_size: int = MAX_VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = DROPOUT
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_length, max_seq_length) * float('-inf'), diagonal=1)
        )
        
        # Store activations for visualization
        self.last_layer_activations = []
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, x: torch.Tensor, store_activations: bool = False) -> torch.Tensor:
        """Forward pass."""
        seq_len = x.size(1)
        
        # Embed words
        x = self.word_embedding(x)
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        mask = self.causal_mask[:seq_len, :seq_len]
        
        if store_activations:
            self.last_layer_activations = []
        
        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)
            if store_activations:
                self.last_layer_activations.append(x.detach().mean(dim=-1).squeeze().tolist())
        
        logits = self.output(x)
        return logits
    
    def get_next_word_probs(
        self, 
        word_indices: List[int], 
        vocab: WordVocabulary,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Get probability distribution for next word."""
        self.eval()
        
        if not word_indices:
            word_indices = [vocab.get_idx(PAD_TOKEN)]
        
        # Truncate to max length
        word_indices = word_indices[-self.max_seq_length:]
        x = torch.tensor([word_indices], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.forward(x)
            last_logits = logits[0, -1, :] / temperature
            
            # Mask out special tokens (except EOS)
            last_logits[vocab.get_idx(PAD_TOKEN)] = float('-inf')
            last_logits[vocab.get_idx(UNK_TOKEN)] = float('-inf')
            
            probs = F.softmax(last_logits, dim=-1)
        
        # Get top words
        vocab_size = len(vocab)
        top_k = min(20, vocab_size)
        top_probs, top_indices = torch.topk(probs[:vocab_size], top_k)
        
        prob_dict = {
            vocab.get_word(idx.item()): prob.item()
            for idx, prob in zip(top_indices, top_probs)
        }
        
        return probs, prob_dict
    
    def sample_next_word(
        self, 
        word_indices: List[int], 
        vocab: WordVocabulary,
        temperature: float = 1.0
    ) -> Tuple[str, int, float]:
        """Sample the next word."""
        probs, _ = self.get_next_word_probs(word_indices, vocab, temperature)
        
        # Only sample from actual vocabulary indices (not the full MAX_VOCAB_SIZE)
        vocab_size = len(vocab)
        valid_probs = probs[:vocab_size].clone()
        
        # Explicitly zero out special tokens to prevent them from being sampled
        valid_probs[vocab.get_idx(PAD_TOKEN)] = 0
        valid_probs[vocab.get_idx(UNK_TOKEN)] = 0
        valid_probs = valid_probs / valid_probs.sum()  # Renormalize
        
        idx = torch.multinomial(valid_probs, 1).item()
        word = vocab.get_word(idx)
        log_prob = torch.log(valid_probs[idx] + 1e-10).item()
        
        return word, idx, log_prob
    
    def get_visualization_data(self, word_indices: List[int], vocab: WordVocabulary) -> Dict:
        """Get visualization data for the neural network."""
        self.eval()
        
        if not word_indices:
            word_indices = [vocab.get_idx(PAD_TOKEN)]
        
        word_indices = word_indices[-self.max_seq_length:]
        x = torch.tensor([word_indices], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.forward(x, store_activations=True)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            # Top words
            vocab_size = len(vocab)
            top_k = min(10, vocab_size)
            top_probs, top_indices = torch.topk(probs[:vocab_size], top_k)
            word_probs = [
                {"word": vocab.get_word(idx.item()), "prob": prob.item()}
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            # Attention (pseudo - based on embedding similarity)
            embeddings = self.word_embedding(x)
            last_embedding = embeddings[0, -1, :]
            attention_scores = F.cosine_similarity(
                embeddings[0], 
                last_embedding.unsqueeze(0), 
                dim=-1
            )
            attention = F.softmax(attention_scores, dim=0).tolist()
            
            # Layer activations
            layer_activations = []
            for acts in self.last_layer_activations:
                if isinstance(acts, list):
                    acts_array = np.array(acts)
                    if acts_array.max() != acts_array.min():
                        normalized = ((acts_array - acts_array.min()) / (acts_array.max() - acts_array.min())).tolist()
                    else:
                        normalized = [0.5] * len(acts)
                    layer_activations.append(normalized)
        
        # Get words from indices
        context_words = [vocab.get_word(idx) for idx in word_indices]
        
        return {
            "word_probs": word_probs,
            "attention": attention,
            "layer_activations": layer_activations,
            "context_words": context_words,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads
        }


class WordAgent:
    """An agent that generates text word by word."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        model: Optional[WordTransformer] = None,
        exploration_rate: float = 0.5,
        shared_vocabulary: Optional[Set[str]] = None,
        learning_rate: float = 0.001,
        target_length: int = 15,  # Max words per turn (no curriculum)
        use_10k: bool = False  # Whether to use 10k common words
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.use_10k = use_10k  # Store for reference
        
        # Vocabulary: seed + (10k common words OR discovered words)
        self.vocab = WordVocabulary(shared_vocabulary, use_10k=use_10k)
        
        # Model (uses vocab size)
        self.model = model or WordTransformer(vocab_size=MAX_VOCAB_SIZE)
        self.exploration_rate = exploration_rate
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Current state
        self.word_indices: List[int] = []
        self.typed_text: str = ""
        self.log_probs: List[float] = []
        self.score: float = 0.0
        self.scores_breakdown: Dict[str, float] = {}
        
        self.temperature = 1.0
        
        # Word count limit per turn
        self.target_length = target_length  # max words per turn
        
        # Conversation context (for multi-turn)
        self.context_words: List[str] = []  # words from GPT responses
    
    def reset(self, keep_context: bool = False):
        """Reset for new generation.
        
        Args:
            keep_context: If True, preserve conversation context for multi-turn
        """
        self.word_indices = []
        self.typed_text = ""
        self.log_probs = []
        self.score = 0.0
        self.scores_breakdown = {}
        if not keep_context:
            self.context_words = []
    
    def update_vocabulary(self, words: Set[str]):
        """Update vocabulary with new discovered words."""
        for word in words:
            self.vocab.add_word(word)
    
    def add_words(self, words: List[str]):
        """Add discovered words (compatibility method)."""
        for word in words:
            self.vocab.add_word(word)
    
    @property
    def vocabulary(self) -> Set[str]:
        """Get vocabulary as set (compatibility)."""
        return set(self.vocab.get_all_words())
    
    def step(self, memory_bias: Optional[Dict[str, float]] = None) -> str:
        """Generate the next word.
        
        Uses conversation context if available.
        """
        # Check word count limit
        current_word_count = len(self.typed_text.split()) if self.typed_text else 0
        if current_word_count >= self.target_length:
            return ""  # Stop generation (max words reached)
        
        self.model.train()
        
        # Build context: conversation history + current generation
        context_indices = []
        if self.context_words:
            # Add context words from GPT responses
            for w in self.context_words[-10:]:  # Last 10 context words
                idx = self.vocab.get_idx(w.lower())
                if idx != self.vocab.get_idx(UNK_TOKEN):
                    context_indices.append(idx)
        context_indices.extend(self.word_indices)
        
        # Exploration: random word from vocabulary
        if np.random.random() < self.exploration_rate:
            available_words = self.vocab.get_all_words()
            if available_words:
                word = np.random.choice(available_words)
                idx = self.vocab.get_idx(word)
                self.word_indices.append(idx)
                self.typed_text += word + " "
                self.log_probs.append(0.0)
                return word + " "
        
        # Exploitation: sample from model (using context)
        word, idx, log_prob = self.model.sample_next_word(
            context_indices if context_indices else self.word_indices, 
            self.vocab, 
            self.temperature
        )
        
        # Check for EOS
        if word == EOS_TOKEN:
            return ""
        
        self.word_indices.append(idx)
        self.typed_text += word + " "
        self.log_probs.append(log_prob)
        
        return word + " "
    
    def set_context(self, gpt_response: str):
        """Set conversation context from GPT response.
        
        Extracts words from GPT response to influence next generation.
        """
        if not gpt_response:
            return
        # Extract words, filter to vocabulary
        words = gpt_response.lower().split()
        self.context_words = [w.strip('.,!?') for w in words if w.strip('.,!?')]
    
    def imitation_step(self, target_word: str) -> float:
        """Perform an imitation learning step.
        
        Returns log probability of target word given current context.
        """
        self.model.train()
        
        # Get index of target word
        target_idx = self.vocab.get_idx(target_word.lower())
        if target_idx == self.vocab.get_idx(UNK_TOKEN):
            return 0.0  # Skip unknown words
        
        # Forward pass
        context = self.word_indices if self.word_indices else [self.vocab.get_idx(PAD_TOKEN)]
        x = torch.tensor([context[-32:]], dtype=torch.long)  # Use last 32 words
        
        with torch.enable_grad():
            logits = self.model(x)
            last_logits = logits[0, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)
            target_log_prob = log_probs[target_idx].item()
        
        # Add to current generation
        self.word_indices.append(target_idx)
        self.typed_text += target_word + " "
        self.log_probs.append(target_log_prob)
        
        return target_log_prob
    
    def train_on_sequence(self, word_sequence: List[str], learning_rate: float = 0.001):
        """Train the model on a target word sequence (imitation learning).
        
        Uses cross-entropy loss to learn from GPT patterns.
        """
        if len(word_sequence) < 2:
            return
        
        # Convert words to indices
        indices = []
        for word in word_sequence:
            idx = self.vocab.get_idx(word.lower())
            if idx != self.vocab.get_idx(UNK_TOKEN):
                indices.append(idx)
        
        if len(indices) < 2:
            return
        
        # Prepare input/target pairs
        x = torch.tensor([indices[:-1]], dtype=torch.long)
        targets = torch.tensor([indices[1:]], dtype=torch.long)
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(x)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.vocab.get_idx(PAD_TOKEN)
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
    
    def update(self, reward: float, baseline: float = 0.0):
        """Update model with REINFORCE."""
        if not self.log_probs:
            return
        
        advantage = reward - baseline
        valid_log_probs = [lp for lp in self.log_probs if lp != 0.0]
        
        if valid_log_probs:
            self._pseudo_gradient_update(advantage)
    
    def _pseudo_gradient_update(self, advantage: float):
        """Apply pseudo-gradient update."""
        lr_scale = 0.01 * advantage
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * 0.001 * advantage
                param.add_(noise)
    
    def set_score(self, score: float, breakdown: Dict[str, float]):
        """Set agent's score."""
        self.score = score
        self.scores_breakdown = breakdown
    
    def decay_exploration(self, decay_rate: float = 0.99, min_exploration: float = 0.02):
        """Decay exploration rate."""
        self.exploration_rate = max(min_exploration, self.exploration_rate * decay_rate)
    
    def get_visualization_data(self) -> Dict:
        """Get visualization data."""
        return self.model.get_visualization_data(self.word_indices, self.vocab)
    
    def to_dict(self, include_viz: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "typed_text": self.typed_text,
            "score": self.score,
            "scores_breakdown": self.scores_breakdown,
            "exploration_rate": self.exploration_rate,
            "vocabulary": list(self.vocabulary),
            "vocabulary_size": len(self.vocabulary)
        }
        if include_viz:
            data["viz"] = self.get_visualization_data()
        return data
    
    def save(self, path: str):
        """Save agent state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_path = path + ".pt"
        torch.save(self.model.state_dict(), model_path)
        
        meta_path = path + ".json"
        meta = {
            "id": self.id,
            "exploration_rate": self.exploration_rate,
            "temperature": self.temperature,
            "vocab_words": self.vocab.get_all_words()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    @classmethod
    def load(cls, path: str, shared_vocabulary: Optional[Set[str]] = None) -> 'WordAgent':
        """Load agent state."""
        meta_path = path + ".json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Combine saved vocab with shared
        all_words = set(meta.get("vocab_words", []))
        if shared_vocabulary:
            all_words.update(shared_vocabulary)
        
        agent = cls(
            agent_id=meta["id"],
            exploration_rate=meta["exploration_rate"],
            shared_vocabulary=all_words
        )
        agent.temperature = meta.get("temperature", 1.0)
        
        model_path = path + ".pt"
        if os.path.exists(model_path):
            agent.model.load_state_dict(torch.load(model_path, weights_only=True))
        
        return agent


def create_word_agent_pool(
    num_agents: int = 12, 
    shared_vocabulary: Optional[Set[str]] = None,
    load_existing: bool = True,
    use_10k: bool = False
) -> List[WordAgent]:
    """Create a pool of word agents.
    
    Args:
        num_agents: Number of agents to create.
        shared_vocabulary: Custom discovered vocabulary to use.
        load_existing: Whether to load existing saved agents.
        use_10k: If True, use 10k common words. If False, use seed + shared_vocabulary.
    """
    agents_dir = os.path.join(os.path.dirname(__file__), "data", "word_agents")
    agents = []
    
    # Only load existing if not changing vocab mode (to ensure consistent vocab size)
    if load_existing and os.path.exists(agents_dir) and not use_10k:
        existing_files = [f for f in os.listdir(agents_dir) if f.endswith('.json')]
        for agent_file in existing_files[:num_agents]:
            agent_id = agent_file.replace('.json', '')
            try:
                agent = WordAgent.load(
                    os.path.join(agents_dir, agent_id),
                    shared_vocabulary
                )
                agents.append(agent)
                print(f"Loaded word agent {agent_id}")
            except Exception as e:
                print(f"Failed to load word agent {agent_id}: {e}")
    
    # Create new agents if needed
    vocab_for_new = None if use_10k else shared_vocabulary
    print(f"[WORD_AGENT] Creating pool with use_10k={use_10k}, vocab size={len(shared_vocabulary) if shared_vocabulary else 0}")
    
    while len(agents) < num_agents:
        agent = WordAgent(shared_vocabulary=vocab_for_new, use_10k=use_10k)
        agents.append(agent)
        print(f"[WORD_AGENT] Created agent with vocab size {len(agent.vocab)}")
    
    return agents


def save_word_agents(agents: List[WordAgent]):
    """Save all word agents."""
    agents_dir = os.path.join(os.path.dirname(__file__), "data", "word_agents")
    os.makedirs(agents_dir, exist_ok=True)
    
    for agent in agents:
        path = os.path.join(agents_dir, agent.id)
        agent.save(path)

