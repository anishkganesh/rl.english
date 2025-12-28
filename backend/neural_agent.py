"""Neural network agent using a small transformer for character generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
import uuid
import math


# Character vocabulary
CHARS = list("abcdefghijklmnopqrstuvwxyz .,!?")
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(CHARS)

# Model hyperparameters
CONTEXT_LENGTH = 64
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# Space bias - boost space probability to encourage word separation
SPACE_BIAS = 3.0  # Multiply space logit by this factor
SPACE_IDX = CHAR_TO_IDX[' ']

# Random exploration also needs space bias - 25% chance to type space
EXPLORATION_SPACE_PROB = 0.25


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CharTransformer(nn.Module):
    """Small transformer for character-level text generation."""
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        context_length: int = CONTEXT_LENGTH,
        dropout: float = DROPOUT
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_length = context_length
        
        # Embedding layers
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, context_length)
        
        # Transformer encoder layers (with attention output)
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
        
        # Causal mask for autoregressive generation
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(context_length, context_length) * float('-inf'), diagonal=1)
        )
        
        # Store activations for visualization
        self.last_attention_weights = None
        self.last_layer_activations = []
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, x: torch.Tensor, store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len] with character indices
            store_activations: Whether to store layer activations for visualization
            
        Returns:
            Logits of shape [batch, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        
        # Embed characters
        x = self.char_embedding(x)  # [batch, seq_len, embed_dim]
        x = self.pos_encoding(x)
        
        # Apply transformer layers with causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        
        if store_activations:
            self.last_layer_activations = []
        
        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)
            if store_activations:
                # Store mean activation per position
                self.last_layer_activations.append(x.detach().mean(dim=-1).squeeze().tolist())
        
        # Project to vocabulary
        logits = self.output(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_next_char_probs(self, context: str, temperature: float = 1.0, apply_space_bias: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get probability distribution for next character.
        
        Args:
            context: String of previous characters
            temperature: Sampling temperature (higher = more random)
            apply_space_bias: Whether to boost space probability
            
        Returns:
            (probabilities tensor, dict mapping chars to probs)
        """
        self.eval()
        
        # Convert context to indices
        if not context:
            # Start with space
            indices = [CHAR_TO_IDX.get(' ', 0)]
        else:
            indices = [CHAR_TO_IDX.get(c, CHAR_TO_IDX[' ']) for c in context[-self.context_length:]]
        
        x = torch.tensor([indices], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.forward(x)
            # Get logits for the last position
            last_logits = logits[0, -1, :] / temperature
            
            # Apply space bias to encourage word separation
            if apply_space_bias:
                last_logits[SPACE_IDX] = last_logits[SPACE_IDX] * SPACE_BIAS
            
            probs = F.softmax(last_logits, dim=-1)
        
        # Convert to dict
        prob_dict = {IDX_TO_CHAR[i]: probs[i].item() for i in range(VOCAB_SIZE)}
        
        return probs, prob_dict
    
    def sample_next_char(self, context: str, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Sample the next character.
        
        Returns:
            (sampled character, log probability)
        """
        probs, _ = self.get_next_char_probs(context, temperature)
        
        # Sample from distribution
        idx = torch.multinomial(probs, 1).item()
        char = IDX_TO_CHAR[idx]
        log_prob = torch.log(probs[idx] + 1e-10).item()
        
        return char, log_prob
    
    def generate(self, max_length: int = 50, temperature: float = 1.0) -> Tuple[str, List[float]]:
        """
        Generate a string of characters.
        
        Returns:
            (generated text, list of log probabilities for each char)
        """
        self.eval()
        text = ""
        log_probs = []
        
        for _ in range(max_length):
            char, log_prob = self.sample_next_char(text, temperature)
            text += char
            log_probs.append(log_prob)
        
        return text, log_probs
    
    def get_visualization_data(self, context: str) -> Dict:
        """
        Get visualization data for the neural network.
        
        Returns:
            Dict with attention, probabilities, and layer activations
        """
        self.eval()
        
        # Convert context to indices
        if not context:
            indices = [CHAR_TO_IDX.get(' ', 0)]
            context = ' '
        else:
            indices = [CHAR_TO_IDX.get(c, CHAR_TO_IDX[' ']) for c in context[-self.context_length:]]
        
        x = torch.tensor([indices], dtype=torch.long)
        
        with torch.no_grad():
            # Forward pass with activation storage
            logits = self.forward(x, store_activations=True)
            
            # Get probabilities for next character
            last_logits = logits[0, -1, :]
            last_logits[SPACE_IDX] = last_logits[SPACE_IDX] * SPACE_BIAS
            probs = F.softmax(last_logits, dim=-1)
            
            # Get top 10 character probabilities
            top_probs, top_indices = torch.topk(probs, min(10, VOCAB_SIZE))
            char_probs = [
                {"char": IDX_TO_CHAR[idx.item()], "prob": prob.item()}
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            # Compute pseudo-attention based on embedding similarity
            # This shows which characters in context influenced the prediction
            embeddings = self.char_embedding(x)  # [1, seq_len, embed_dim]
            last_embedding = embeddings[0, -1, :]  # [embed_dim]
            
            # Cosine similarity between last position and all positions
            attention_scores = F.cosine_similarity(
                embeddings[0], 
                last_embedding.unsqueeze(0), 
                dim=-1
            )
            attention = F.softmax(attention_scores, dim=0).tolist()
            
            # Layer activations (normalized)
            layer_activations = []
            for i, acts in enumerate(self.last_layer_activations):
                if isinstance(acts, list):
                    # Normalize to 0-1 range
                    acts_array = np.array(acts)
                    if acts_array.max() != acts_array.min():
                        normalized = ((acts_array - acts_array.min()) / (acts_array.max() - acts_array.min())).tolist()
                    else:
                        normalized = [0.5] * len(acts)
                    layer_activations.append(normalized)
            
        return {
            "char_probs": char_probs,
            "attention": attention,
            "layer_activations": layer_activations,
            "context_chars": list(context[-len(attention):]) if attention else [],
            "num_layers": self.num_layers,
            "num_heads": self.num_heads
        }


class NeuralAgent:
    """An agent that uses a neural network to generate text."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        model: Optional[CharTransformer] = None,
        exploration_rate: float = 0.5,
        vocabulary: Optional[Set[str]] = None,
        learning_rate: float = 0.001
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.model = model or CharTransformer()
        self.exploration_rate = exploration_rate
        self.vocabulary: Set[str] = vocabulary or set()
        
        # Optimizer for policy gradient
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Current state
        self.typed_text: str = ""
        self.log_probs: List[float] = []  # Store log probs for REINFORCE
        self.score: float = 0.0
        self.scores_breakdown: Dict[str, float] = {}
        
        # Temperature for sampling (lower = more deterministic)
        self.temperature = 1.0
    
    def reset(self, keep_context: bool = False):
        """Reset the agent for a new generation.
        
        Args:
            keep_context: Ignored for char NN agents (no conversation context)
        """
        self.typed_text = ""
        self.log_probs = []
        self.score = 0.0
        self.scores_breakdown = {}
    
    def add_words(self, words: List[str]):
        """Add discovered words to vocabulary."""
        for word in words:
            if word and len(word) >= 1:
                self.vocabulary.add(word.lower())
    
    def step(self, memory_bias: Optional[Dict[str, float]] = None) -> str:
        """Perform one typing step and return the character(s) typed."""
        self.model.train()  # Enable dropout during generation
        
        # Get context
        context = self.typed_text
        
        # 30% chance to use vocabulary word at word boundaries
        at_word_boundary = not self.typed_text or self.typed_text.endswith(' ')
        if self.vocabulary and at_word_boundary and np.random.random() < 0.3:
            word = np.random.choice(list(self.vocabulary))
            self.typed_text += word + " "
            return word + " "
        
        # Exploration: random character with space bias
        if np.random.random() < self.exploration_rate:
            # Apply space bias during exploration too
            if np.random.random() < EXPLORATION_SPACE_PROB:
                char = ' '
            else:
                # Random letter (excluding space)
                non_space_chars = [c for c in CHARS if c != ' ']
                char = np.random.choice(non_space_chars)
            self.typed_text += char
            self.log_probs.append(0.0)  # No gradient for random actions
            return char
        
        # Exploitation: sample from model
        char, log_prob = self.model.sample_next_char(context, self.temperature)
        self.typed_text += char
        self.log_probs.append(log_prob)
        
        return char
    
    def update(self, reward: float, baseline: float = 0.0):
        """
        Update model weights using REINFORCE policy gradient.
        
        Args:
            reward: The score/reward for this episode
            baseline: Baseline to reduce variance (e.g., average reward)
        """
        if not self.log_probs:
            return
        
        self.optimizer.zero_grad()
        
        # Calculate policy gradient loss
        # loss = -sum(log_prob * (reward - baseline))
        advantage = reward - baseline
        
        # Only use non-zero log probs (from model-sampled actions)
        valid_log_probs = [lp for lp in self.log_probs if lp != 0.0]
        
        if valid_log_probs:
            # Create tensor for backprop
            log_prob_tensor = torch.tensor(valid_log_probs, requires_grad=True)
            loss = -torch.mean(log_prob_tensor) * advantage
            
            # We need to actually backprop through the model
            # This is a simplified version - for proper REINFORCE we'd store the tensors
            # For now, we'll use a pseudo-gradient update
            self._pseudo_gradient_update(advantage)
    
    def _pseudo_gradient_update(self, advantage: float):
        """Apply a pseudo-gradient update based on the advantage."""
        # Scale learning rate by advantage
        lr_scale = 0.01 * advantage
        
        # Update model weights slightly in the direction that produced this output
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.add_(param.grad * lr_scale)
                else:
                    # Small random perturbation scaled by advantage
                    noise = torch.randn_like(param) * 0.001 * advantage
                    param.add_(noise)
    
    def set_score(self, score: float, breakdown: Dict[str, float]):
        """Set the agent's fitness score."""
        self.score = score
        self.scores_breakdown = breakdown
    
    def decay_exploration(self, decay_rate: float = 0.99, min_exploration: float = 0.02):
        """Decay the exploration rate."""
        self.exploration_rate = max(min_exploration, self.exploration_rate * decay_rate)
    
    def get_visualization_data(self) -> Dict:
        """Get neural network visualization data."""
        return self.model.get_visualization_data(self.typed_text)
    
    def to_dict(self, include_viz: bool = False) -> dict:
        """Convert agent state to dictionary for API response."""
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
        """Save agent state and model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model weights
        model_path = path + ".pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata
        meta_path = path + ".json"
        meta = {
            "id": self.id,
            "exploration_rate": self.exploration_rate,
            "vocabulary": list(self.vocabulary),
            "temperature": self.temperature
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    @classmethod
    def load(cls, path: str) -> 'NeuralAgent':
        """Load agent state and model weights."""
        # Load metadata
        meta_path = path + ".json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Create agent with loaded metadata
        agent = cls(
            agent_id=meta["id"],
            exploration_rate=meta["exploration_rate"],
            vocabulary=set(meta.get("vocabulary", []))
        )
        agent.temperature = meta.get("temperature", 1.0)
        
        # Load model weights
        model_path = path + ".pt"
        if os.path.exists(model_path):
            agent.model.load_state_dict(torch.load(model_path, weights_only=True))
        
        return agent


def create_neural_agent_pool(num_agents: int = 12, load_existing: bool = True) -> List[NeuralAgent]:
    """
    Create a pool of neural agents.
    
    Args:
        num_agents: Number of agents to create
        load_existing: Whether to try loading existing saved agents
        
    Returns:
        List of NeuralAgent instances
    """
    agents_dir = os.path.join(os.path.dirname(__file__), "data", "agents")
    agents = []
    
    # Try to load existing agents
    if load_existing and os.path.exists(agents_dir):
        existing_files = [f for f in os.listdir(agents_dir) if f.endswith('.json')]
        for agent_file in existing_files[:num_agents]:
            agent_id = agent_file.replace('.json', '')
            try:
                agent = NeuralAgent.load(os.path.join(agents_dir, agent_id))
                agents.append(agent)
                print(f"Loaded agent {agent_id}")
            except Exception as e:
                print(f"Failed to load agent {agent_id}: {e}")
    
    # Create new agents if needed
    while len(agents) < num_agents:
        agent = NeuralAgent()
        agents.append(agent)
    
    return agents


def save_all_agents(agents: List[NeuralAgent]):
    """Save all agents to disk."""
    agents_dir = os.path.join(os.path.dirname(__file__), "data", "agents")
    os.makedirs(agents_dir, exist_ok=True)
    
    for agent in agents:
        path = os.path.join(agents_dir, agent.id)
        agent.save(path)

