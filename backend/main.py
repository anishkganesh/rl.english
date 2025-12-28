"""FastAPI application with WebSocket for real-time agent communication."""

import asyncio
import json
import time
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from config import config
from agents import Agent, create_agent_pool, AgentGenome
from neural_agent import NeuralAgent, create_neural_agent_pool, save_all_agents
from word_agent import WordAgent, create_word_agent_pool, save_word_agents
from word_genome_agent import (
    WordGenomeAgent, create_word_genome_agent_pool, evolve_word_genome_generation,
    save_word_genome_agents, learn_from_success_word_genome
)
from evolution import evolve_generation
from scorer import (
    score_conversation, score_vocabulary, score_structure_curriculum,
    score_word_genome_output, score_batch_openai, score_word_genome_openai
)
from memory import pattern_memory

# OpenAI client for conversation
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None


async def get_gpt_response(agent_text: str, conversation_history: List[Dict] = None) -> str:
    """Get a conversational response from GPT for multi-turn dialogue.
    
    Args:
        agent_text: The agent's message
        conversation_history: Previous turns in the conversation
    
    Returns:
        GPT's response text
    """
    if not openai_client:
        return ""
    
    if not agent_text or len(agent_text.strip()) < 2:
        return ""
    
    try:
        # Build conversation messages
        messages = [
            {"role": "system", "content": "You are a friendly, patient English teacher having a conversation with a student who is just learning to communicate. Respond simply and naturally. Keep responses under 20 words."}
        ]
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-3:]:  # Last 3 turns
                if turn.get("agent_text"):
                    messages.append({"role": "user", "content": turn["agent_text"]})
                if turn.get("openai_response"):
                    messages.append({"role": "assistant", "content": turn["openai_response"]})
        
        # Add current message
        messages.append({"role": "user", "content": agent_text})
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT response failed: {e}")
        return ""


def extract_words_from_response(response: str) -> List[str]:
    """Extract words from a GPT response for imitation learning."""
    if not response:
        return []
    # Clean and extract words
    words = response.lower().split()
    return [w.strip('.,!?;:"\'') for w in words if w.strip('.,!?;:"\'')]


# Simple prompts for GPT-first mode (Phase 4)
GPT_FIRST_PROMPTS = [
    "Hello! How are you today?",
    "Hi there! What's your name?",
    "Good morning! Nice to meet you.",
    "Hello! Do you like to read?",
    "Hi! What do you like to do?",
    "Hello! What is your favorite color?",
    "Good day! How can I help you?",
    "Hi! Do you have any questions?",
    "Hello! Tell me about yourself.",
    "Hi there! What are you thinking about?",
]


async def generate_gpt_first_message() -> str:
    """Generate a simple first message from GPT to start conversation in Phase 4.
    
    Returns a friendly, simple prompt that the agent should respond to.
    """
    if openai_client:
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate a simple, friendly greeting or question to start a conversation with someone learning English. Keep it under 10 words. Examples: 'Hello! How are you?', 'Hi! What is your name?', 'Good morning! Nice to meet you.'"},
                    {"role": "user", "content": "Generate a greeting:"}
                ],
                temperature=0.9,  # High variation
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] GPT first message failed: {e}")
    
    # Fallback to pre-defined prompts
    return random.choice(GPT_FIRST_PROMPTS)


class ModelType(str, Enum):
    GENOME = "genome"
    CHAR_NN = "char_nn"
    WORD_NN = "word_nn"
    WORD_GENOME = "word_genome"  # Word-level genetic algorithm
    CONCURRENT = "concurrent"  # Runs Char NN and Word NN simultaneously


class WordGenomePhase(int, Enum):
    """Curriculum learning phases for Word Genome."""
    STRUCTURE = 1      # Learn capitalization, punctuation, spacing
    PATTERNS = 2       # Learn bigrams, subject-verb patterns
    SENTENCES = 3      # Full sentence generation with SVO
    CONVERSATION = 4   # GPT-first dialogue, response relevance


# Phase mastery thresholds
PHASE_MASTERY_THRESHOLD = 0.80  # 80% of agents must achieve phase goals
PHASE_HISTORY_LENGTH = 5        # Check last 5 generations for mastery


# Agent type alias
AgentUnion = Union[Agent, NeuralAgent, WordAgent, WordGenomeAgent]

# Persistence paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
INTERESTING_CONVS_FILE = DATA_DIR / "interesting_conversations.json"
BEST_AGENTS_FILE = DATA_DIR / "best_agents.json"
DISCOVERED_BIGRAMS_FILE = DATA_DIR / "discovered_bigrams.json"
DISCOVERED_TRIGRAMS_FILE = DATA_DIR / "discovered_trigrams.json"
DISCOVERED_SENTENCES_FILE = DATA_DIR / "discovered_sentences.json"


def save_to_json(file_path: Path, data: List[Dict]):
    """Save data to JSON file."""
    existing = []
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                existing = json.load(f)
        except:
            existing = []
    
    existing.extend(data)
    # Keep last 1000 entries
    existing = existing[-1000:]
    
    with open(file_path, 'w') as f:
        json.dump(existing, f, indent=2)


def load_from_json(file_path: Path) -> List[Dict]:
    """Load data from JSON file."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_set_to_json(file_path: Path, data: set):
    """Save a set to JSON file (converts to list)."""
    with open(file_path, 'w') as f:
        json.dump(list(data), f, indent=2)


def load_set_from_json(file_path: Path) -> set:
    """Load a set from JSON file."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_patterns(state: 'TrainingState'):
    """Save all discovered patterns to JSON files."""
    save_set_to_json(DISCOVERED_BIGRAMS_FILE, state.discovered_bigrams)
    save_set_to_json(DISCOVERED_TRIGRAMS_FILE, state.discovered_trigrams)
    
    # Save sentences (list of dicts, not set)
    with open(DISCOVERED_SENTENCES_FILE, 'w') as f:
        json.dump(state.discovered_sentences[:config.max_discovered_sentences], f, indent=2)


def load_patterns(state: 'TrainingState'):
    """Load discovered patterns from JSON files."""
    state.discovered_bigrams = load_set_from_json(DISCOVERED_BIGRAMS_FILE)
    state.discovered_trigrams = load_set_from_json(DISCOVERED_TRIGRAMS_FILE)
    
    if DISCOVERED_SENTENCES_FILE.exists():
        try:
            with open(DISCOVERED_SENTENCES_FILE, 'r') as f:
                state.discovered_sentences = json.load(f)
        except:
            state.discovered_sentences = []


def is_interesting_conversation(conversation: List[Dict], scores: Dict) -> bool:
    """Check if a conversation is interesting enough to save.
    
    Criteria:
    - Has at least one valid English word (vocabulary score > 0)
    - GPT response is not ALL "I don't understand" type messages
    """
    if not conversation:
        return False
    
    # Check vocabulary details - even one valid word counts!
    # Valid words are stored directly in details.valid_words as a list
    words_found = scores.get("details", {}).get("valid_words", [])
    
    # If there's at least one valid English word
    if len(words_found) > 0:
        # Check if GPT engaged meaningfully (at least one non-confused response)
        confusion_patterns = [
            "don't understand",
            "unclear",
            "typo",
            "jumbled",
            "random letters",
            "mix-up",
            "clarify what you mean",
            "might be a typo",
            "might have been a typo",
            "bit of a jumble"
        ]
        
        # Count how many turns have confused GPT responses
        confused_turns = 0
        for turn in conversation:
            gpt_response = turn.get("openai_response", "").lower()
            if any(pattern in gpt_response for pattern in confusion_patterns):
                confused_turns += 1
        
        # Save if at least one turn was not confused
        if confused_turns < len(conversation):
            return True
    
    # Also save if total score is decent (even if no complete words)
    total_score = scores.get("total", 0)
    if total_score > 0.3:
        return True
    
    return False


class Milestone:
    """Track achievement milestones."""
    
    def __init__(self):
        self.first_word = False
        self.first_sentence = False  # Grammatically correct
        self.first_coherent = False  # Coherence > 0.7
        self.sustained_conversation = False  # 3 turns with grammar > 0.8
        self.achieved = []
    
    def check(self, scores: Dict) -> List[str]:
        """Check for new milestones, return list of newly achieved."""
        new_achievements = []
        
        vocab_details = scores.get("details", {}).get("vocabulary", {})
        grammar_details = scores.get("details", {}).get("grammar", {})
        
        # First valid English word
        if not self.first_word and vocab_details.get("valid_words", 0) > 0:
            self.first_word = True
            new_achievements.append("first_word")
            self.achieved.append({"type": "first_word", "text": vocab_details.get("words_found", [])[:3]})
        
        # First grammatically correct sentence
        if not self.first_sentence and grammar_details.get("is_grammatical", False):
            self.first_sentence = True
            new_achievements.append("first_sentence")
            self.achieved.append({"type": "first_sentence"})
        
        # First coherent exchange
        if not self.first_coherent and scores.get("coherence_score", 0) > 0.7:
            self.first_coherent = True
            new_achievements.append("first_coherent")
            self.achieved.append({"type": "first_coherent"})
        
        return new_achievements
    
    def to_dict(self) -> Dict:
        return {
            "first_word": self.first_word,
            "first_sentence": self.first_sentence,
            "first_coherent": self.first_coherent,
            "sustained_conversation": self.sustained_conversation,
            "achieved": self.achieved
        }
    
    def reset(self):
        self.__init__()


class ModelState:
    """State for a specific model type."""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.agents: List[AgentUnion] = []
        self.generation: int = 0
        self.history: List[Dict] = []
        self.best_outputs: List[Dict] = []
        self.milestones = Milestone()
        self.current_conversations: Dict[str, List[Dict]] = {}
        self.baseline_reward: float = 0.0
        # Per-model running state
        self.is_running: bool = False
        self.task: Optional[asyncio.Task] = None
        
        # Word Genome curriculum learning state
        self.current_phase: WordGenomePhase = WordGenomePhase.STRUCTURE
        self.phase_progress: float = 0.0  # 0-1 progress toward next phase
        self.phase_history: List[Dict] = []  # Track phase metrics over generations
        self.gpt_first_message: str = ""  # GPT's message for Phase 4
    
    def reset(self, keep_agents: bool = False):
        """Reset this model's state."""
        self.generation = 0
        self.history = []
        self.best_outputs = []
        self.current_conversations = {}
        self.milestones.reset()
        self.baseline_reward = 0.0
        # Reset phase tracking
        self.current_phase = WordGenomePhase.STRUCTURE
        self.phase_progress = 0.0
        self.phase_history = []
        self.gpt_first_message = ""


class TrainingState:
    """Global training state manager with multi-model support."""
    
    def __init__(self):
        # Separate state for each model type
        self.model_states: Dict[ModelType, ModelState] = {
            ModelType.GENOME: ModelState(ModelType.GENOME),
            ModelType.CHAR_NN: ModelState(ModelType.CHAR_NN),
            ModelType.WORD_NN: ModelState(ModelType.WORD_NN),
            ModelType.WORD_GENOME: ModelState(ModelType.WORD_GENOME),
        }
        
        # Current active model (for view)
        self.current_model: ModelType = ModelType.CHAR_NN
        
        # Shared state
        self.connected_clients: List[WebSocket] = []
        self.global_vocabulary: set = set()  # Shared across all models
        
        # Pattern discovery banks (shared across all models)
        self.discovered_bigrams: set = set()  # e.g., "i am", "you are"
        self.discovered_trigrams: set = set()  # e.g., "how are you"
        self.discovered_sentences: List[Dict] = []  # {text, score, agent_id, generation, model_type}
        
        # Toggle for using 10k words vs discovered vocabulary
        self.use_10k_words: bool = True
    
    def is_model_running(self, model_type: ModelType) -> bool:
        """Check if a specific model is running."""
        if model_type == ModelType.CONCURRENT:
            return (self.model_states[ModelType.CHAR_NN].is_running or 
                    self.model_states[ModelType.WORD_NN].is_running)
        return self.model_states[model_type].is_running
    
    def any_model_running(self) -> bool:
        """Check if any model is running."""
        return any(ms.is_running for ms in self.model_states.values())
    
    @property
    def is_running(self) -> bool:
        """Legacy property - returns if current model is running."""
        return self.is_model_running(self.current_model)
    
    @property
    def current_state(self) -> ModelState:
        """Get current model's state. Returns CHAR_NN state for concurrent mode."""
        if self.current_model == ModelType.CONCURRENT:
            # For concurrent mode, return char_nn state as a fallback
            return self.model_states[ModelType.CHAR_NN]
        return self.model_states[self.current_model]
    
    @property
    def agents(self) -> List[AgentUnion]:
        """Get current model's agents."""
        if self.current_model == ModelType.CONCURRENT:
            # Return combined agents for concurrent mode
            return (self.model_states[ModelType.CHAR_NN].agents[:6] + 
                    self.model_states[ModelType.WORD_NN].agents[:6])
        return self.current_state.agents
    
    @agents.setter
    def agents(self, value: List[AgentUnion]):
        if self.current_model != ModelType.CONCURRENT:
            self.current_state.agents = value
    
    @property
    def generation(self) -> int:
        if self.current_model == ModelType.CONCURRENT:
            return self.model_states[ModelType.CHAR_NN].generation
        return self.current_state.generation
    
    @generation.setter
    def generation(self, value: int):
        if self.current_model != ModelType.CONCURRENT:
            self.current_state.generation = value
    
    @property
    def history(self) -> List[Dict]:
        if self.current_model == ModelType.CONCURRENT:
            return self.model_states[ModelType.CHAR_NN].history
        return self.current_state.history
    
    @property
    def best_outputs(self) -> List[Dict]:
        if self.current_model == ModelType.CONCURRENT:
            return self.model_states[ModelType.CHAR_NN].best_outputs
        return self.current_state.best_outputs
    
    @best_outputs.setter
    def best_outputs(self, value: List[Dict]):
        if self.current_model != ModelType.CONCURRENT:
            self.current_state.best_outputs = value
    
    @property
    def milestones(self) -> Milestone:
        if self.current_model == ModelType.CONCURRENT:
            return self.model_states[ModelType.CHAR_NN].milestones
        return self.current_state.milestones
    
    @property
    def current_conversations(self) -> Dict[str, List[Dict]]:
        if self.current_model == ModelType.CONCURRENT:
            return self.model_states[ModelType.CHAR_NN].current_conversations
        return self.current_state.current_conversations
    
    @current_conversations.setter
    def current_conversations(self, value: Dict[str, List[Dict]]):
        if self.current_model != ModelType.CONCURRENT:
            self.current_state.current_conversations = value
    
    @property
    def baseline_reward(self) -> float:
        if self.current_model == ModelType.CONCURRENT:
            return self.model_states[ModelType.CHAR_NN].baseline_reward
        return self.current_state.baseline_reward
    
    @baseline_reward.setter
    def baseline_reward(self, value: float):
        self.current_state.baseline_reward = value
    
    def initialize(self, load_existing: bool = True):
        """Initialize all model types."""
        # Initialize Genome agents
        self.model_states[ModelType.GENOME].agents = create_agent_pool(config.num_agents)
        
        # Initialize Char NN agents
        self.model_states[ModelType.CHAR_NN].agents = create_neural_agent_pool(
            config.num_agents, load_existing=load_existing
        )
        
        # Initialize Word NN agents (with shared vocabulary)
        self.model_states[ModelType.WORD_NN].agents = create_word_agent_pool(
            config.num_agents, 
            shared_vocabulary=self.global_vocabulary,
            load_existing=load_existing
        )
        
        # Initialize Word Genome agents (word-level genetic algorithm)
        # Use global vocab if use_10k_words is False
        word_genome_vocab = None if self.use_10k_words else self.global_vocabulary
        self.model_states[ModelType.WORD_GENOME].agents = create_word_genome_agent_pool(
            config.num_agents,
            custom_vocab=word_genome_vocab
        )
        
        # Don't automatically collect vocabulary from loaded agents
        # Vocabulary is now managed separately and only grows from new discoveries
        # This prevents old saved agents from polluting the global vocabulary
        pass
    
    def switch_model(self, model_type: ModelType):
        """Switch to a different model type."""
        self.current_model = model_type
        
        # Update word agents with current global vocabulary
        if model_type == ModelType.WORD_NN or model_type == ModelType.CONCURRENT:
            for agent in self.model_states[ModelType.WORD_NN].agents:
                agent.update_vocabulary(self.global_vocabulary)
    
    def reset(self, keep_agents: bool = False):
        """Reset current model's state.
        
        Args:
            keep_agents: If True, keep trained agents. If False, create new ones.
        """
        self.current_state.reset(keep_agents)
        
        if not keep_agents:
            if self.current_model == ModelType.GENOME:
                self.current_state.agents = create_agent_pool(config.num_agents)
            elif self.current_model == ModelType.CHAR_NN:
                self.current_state.agents = create_neural_agent_pool(config.num_agents, load_existing=False)
            elif self.current_model == ModelType.WORD_NN:
                self.current_state.agents = create_word_agent_pool(
                    config.num_agents, 
                    shared_vocabulary=self.global_vocabulary,
                    load_existing=False
                )
            elif self.current_model == ModelType.WORD_GENOME:
                word_genome_vocab = None if self.use_10k_words else self.global_vocabulary
                self.current_state.agents = create_word_genome_agent_pool(
                    config.num_agents,
                    custom_vocab=word_genome_vocab
                )
            pattern_memory.clear()


state = TrainingState()


# =============================================================================
# Word Genome Phase Transition Logic
# =============================================================================

def check_phase_mastery(model_state: ModelState) -> Dict:
    """Check if current phase goals are met and calculate progress.
    
    Returns:
        Dict with mastery status and progress metrics
    """
    if model_state.model_type != ModelType.WORD_GENOME:
        return {"mastery": False, "progress": 0.0}
    
    history = model_state.phase_history[-PHASE_HISTORY_LENGTH:]
    if not history:
        return {"mastery": False, "progress": 0.0}
    
    phase = model_state.current_phase
    
    if phase == WordGenomePhase.STRUCTURE:
        # Phase 1: 80% of agents have capital start AND punctuation end
        structure_counts = [h.get("structure_mastery", 0) for h in history]
        avg_mastery = sum(structure_counts) / len(structure_counts)
        mastery = avg_mastery >= PHASE_MASTERY_THRESHOLD
        return {"mastery": mastery, "progress": avg_mastery, "metric": "structure"}
    
    elif phase == WordGenomePhase.PATTERNS:
        # Phase 2: 80% of agents produce at least 1 valid bigram
        bigram_counts = [h.get("bigram_mastery", 0) for h in history]
        avg_mastery = sum(bigram_counts) / len(bigram_counts)
        mastery = avg_mastery >= PHASE_MASTERY_THRESHOLD
        return {"mastery": mastery, "progress": avg_mastery, "metric": "bigrams"}
    
    elif phase == WordGenomePhase.SENTENCES:
        # Phase 3: Best agent avg score > 0.6
        best_scores = [h.get("best_score", 0) for h in history]
        avg_best = sum(best_scores) / len(best_scores)
        mastery = avg_best >= 0.6
        return {"mastery": mastery, "progress": min(1.0, avg_best / 0.6), "metric": "sentence_quality"}
    
    else:  # CONVERSATION
        # Phase 4: No automatic progression (end phase)
        return {"mastery": False, "progress": 1.0, "metric": "conversation"}


def transition_phase(model_state: ModelState) -> bool:
    """Check and transition to next phase if mastery achieved.
    
    Returns:
        True if phase transition occurred
    """
    if model_state.model_type != ModelType.WORD_GENOME:
        return False
    
    mastery_check = check_phase_mastery(model_state)
    model_state.phase_progress = mastery_check["progress"]
    
    if mastery_check["mastery"] and model_state.current_phase.value < WordGenomePhase.CONVERSATION.value:
        old_phase = model_state.current_phase
        model_state.current_phase = WordGenomePhase(model_state.current_phase.value + 1)
        model_state.phase_progress = 0.0
        print(f"[PHASE] Transitioned from {old_phase.name} to {model_state.current_phase.name}!")
        return True
    
    return False


def record_phase_metrics(model_state: ModelState, agent_results: List[Dict]):
    """Record metrics for phase transition checking.
    
    Args:
        model_state: The model state to update
        agent_results: List of scoring results for each agent
    """
    if model_state.model_type != ModelType.WORD_GENOME:
        return
    
    # Count structure achievements
    structure_count = sum(
        1 for r in agent_results 
        if r.get("has_capital_start", False) and r.get("has_punctuation_end", False)
    )
    structure_mastery = structure_count / len(agent_results) if agent_results else 0
    
    # Count bigram achievements
    bigram_count = sum(1 for r in agent_results if r.get("bigram_count", 0) >= 1)
    bigram_mastery = bigram_count / len(agent_results) if agent_results else 0
    
    # Get best score
    best_score = max((r.get("total", 0) for r in agent_results), default=0)
    
    model_state.phase_history.append({
        "generation": model_state.generation,
        "structure_mastery": structure_mastery,
        "bigram_mastery": bigram_mastery,
        "best_score": best_score,
        "phase": model_state.current_phase.value
    })
    
    # Keep history bounded
    if len(model_state.phase_history) > 50:
        model_state.phase_history = model_state.phase_history[-50:]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize state on startup - try to load existing agents and patterns."""
    state.initialize(load_existing=True)
    load_patterns(state)  # Load discovered patterns
    print(f"[INIT] Loaded {len(state.discovered_bigrams)} bigrams, {len(state.discovered_trigrams)} trigrams, {len(state.discovered_sentences)} sentences")
    yield
    # Save agents and patterns on shutdown
    try:
        # Save Neural agents
        neural_agents = [a for a in state.model_states[ModelType.CHAR_NN].agents if isinstance(a, NeuralAgent)]
        if neural_agents:
            save_all_agents(neural_agents)
        
        # Save Word agents
        word_agents = [a for a in state.model_states[ModelType.WORD_NN].agents if isinstance(a, WordAgent)]
        if word_agents:
            save_word_agents(word_agents)
        
        # Save Word Genome agents
        word_genome_agents = [a for a in state.model_states[ModelType.WORD_GENOME].agents if isinstance(a, WordGenomeAgent)]
        if word_genome_agents:
            save_word_genome_agents(word_genome_agents)
    except Exception as e:
        print(f"[WARN] Error saving agents: {e}")
    
    save_patterns(state)


app = FastAPI(title="RL English Typing Agents", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint for Railway/deployment
@app.get("/")
async def health_check():
    return {"status": "ok", "service": "rl-english-backend"}


class ConfigUpdate(BaseModel):
    num_agents: Optional[int] = None
    generation_time: Optional[float] = None
    mutation_rate: Optional[float] = None


async def broadcast(message: dict):
    """Send message to all connected WebSocket clients."""
    if not state.connected_clients:
        return
    
    data = json.dumps(message)
    disconnected = []
    
    for client in state.connected_clients:
        try:
            await client.send_text(data)
        except:
            disconnected.append(client)
    
    for client in disconnected:
        state.connected_clients.remove(client)


def detect_sentence_end(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation."""
    if not text:
        return False
    stripped = text.rstrip()
    return stripped and stripped[-1] in '.!?'


async def run_agent_turn(agent: NeuralAgent, context: Optional[str] = None) -> str:
    """Run a single turn for an agent with adaptive timing.
    
    Args:
        agent: The agent to run
        context: Optional conversation context (OpenAI's previous response)
    
    Returns:
        The agent's generated text for this turn
    """
    agent.reset()  # Clear typed text for new turn
    
    start_time = time.time()
    last_output_time = start_time
    
    while True:
        elapsed = time.time() - start_time
        
        # Hard cap at generation_time
        if elapsed >= config.generation_time:
            break
        
        # Check for idle timeout (no output for idle_timeout seconds)
        if time.time() - last_output_time >= config.idle_timeout and agent.typed_text:
            break
        
        # Check for sentence end (adaptive early stopping)
        if detect_sentence_end(agent.typed_text) and len(agent.typed_text) > 5:
            # Give a tiny delay to see if more comes
            await asyncio.sleep(0.1)
            if detect_sentence_end(agent.typed_text):
                break
        
        # Agent types one character using neural network
        char = agent.step()
        
        if char:  # If agent produced output
            last_output_time = time.time()
        
        await asyncio.sleep(config.keystroke_interval)
    
    return agent.typed_text


async def run_conversation_generation(model_type: ModelType = None):
    """Run a full conversation generation with multiple turns.
    
    Args:
        model_type: The model to run generation for. If None, uses state.current_model.
    """
    if model_type is None:
        model_type = state.current_model
    
    model_state = state.model_states[model_type]
    agents = model_state.agents
    
    print(f"[DEBUG] Starting generation {model_state.generation} for model {model_type}")
    
    # Initialize conversations for each agent
    model_state.current_conversations = {a.id: [] for a in agents}
    
    # Include running status for all models
    running_status = {
        "genome": state.model_states[ModelType.GENOME].is_running,
        "char_nn": state.model_states[ModelType.CHAR_NN].is_running,
        "word_nn": state.model_states[ModelType.WORD_NN].is_running
    }
    
    # Broadcast generation start
    # Add phase info for Word Genome
    phase_info = {}
    if model_type == ModelType.WORD_GENOME:
        phase_info = {
            "word_genome_phase": model_state.current_phase.name,
            "word_genome_phase_progress": model_state.phase_progress,
            "word_genome_phase_number": model_state.current_phase.value
        }
    
    await broadcast({
        "type": "generation_start",
        "generation": model_state.generation,
        "agents": [a.to_dict() for a in agents],
        "model_type": model_type.value,
        "running_status": running_status,
        "is_running": model_state.is_running,
        **phase_info
    })
    
    # Determine number of turns based on model type
    if model_type == ModelType.WORD_NN:
        num_turns = config.word_nn_turns_per_gen  # 3 turns for Word NN
    elif model_type == ModelType.WORD_GENOME:
        num_turns = 1  # Single turn for Word Genome (simpler evolution)
    else:
        num_turns = config.turns_per_conversation
    
    # Alternating conversation flow for Word Genome
    # Even generations: GPT speaks first
    # Odd generations: Agent speaks first
    gpt_speaks_first = model_state.generation % 2 == 0
    
    if model_type == ModelType.WORD_GENOME:
        # Set the current phase for all agents (for adaptive sentence length)
        for agent in agents:
            agent.set_phase(model_state.current_phase.value)
        
        # Determine conversation mode based on phase and generation
        if model_state.current_phase.value >= 3:  # Phase 3+ allows alternating
            if gpt_speaks_first:
                gpt_first_msg = await generate_gpt_first_message()
                model_state.gpt_first_message = gpt_first_msg
                
                # Set context for all agents
                for agent in agents:
                    agent.set_context(gpt_first_msg)
                
                # Broadcast GPT's first message
                await broadcast({
                    "type": "gpt_first",
                    "message": gpt_first_msg,
                    "model_type": model_type.value
                })
                
                print(f"[ALTERNATING] Gen {model_state.generation} - GPT first: '{gpt_first_msg}'")
            else:
                # Agent speaks first - clear GPT context
                model_state.gpt_first_message = ""
                print(f"[ALTERNATING] Gen {model_state.generation} - Agent speaks first")
        else:
            # Early phases: simple agent-only generation
            model_state.gpt_first_message = ""
    
    # Run conversation turns
    for turn_num in range(num_turns):
        if not model_state.is_running:
            break
        
        await broadcast({
            "type": "turn_start",
            "turn": turn_num + 1,
            "total_turns": num_turns,
            "model_type": model_type.value
        })
        
        # Reset all agents for this turn (keep context for Word NN/Word Genome multi-turn)
        for agent in agents:
            if model_type in (ModelType.WORD_NN, ModelType.WORD_GENOME) and turn_num > 0:
                agent.reset(keep_context=True)  # Preserve conversation context
            else:
                agent.reset()
        
        # Run all agents in parallel for this turn
        start_time = time.time()
        last_output_times = {a.id: start_time for a in agents}
        finished_agents = set()
        
        while True:
            elapsed = time.time() - start_time
            
            # Hard cap for the turn
            if elapsed >= config.generation_time:
                break
            
            # Check if all agents are finished
            if len(finished_agents) == len(agents):
                break
            
            # Check if training was stopped
            if not model_state.is_running:
                break
            
            # Process all agents in one tick
            updates = []
            for agent in agents:
                if agent.id in finished_agents:
                    continue
                
                # Check idle timeout
                if time.time() - last_output_times[agent.id] >= config.idle_timeout and agent.typed_text:
                    finished_agents.add(agent.id)
                    continue
                
                # Check word count limit for word-level models (generate whole words, not chars)
                if model_type in (ModelType.WORD_NN, ModelType.WORD_GENOME):
                    word_count = len(agent.typed_text.split())
                    if word_count >= config.word_nn_max_length:
                        finished_agents.add(agent.id)
                        continue
                
                # Check sentence end (for char-level models)
                if detect_sentence_end(agent.typed_text) and len(agent.typed_text) > 5:
                    finished_agents.add(agent.id)
                    continue
                
                # Agent types one character using neural network
                char = agent.step()
                
                if char:
                    last_output_times[agent.id] = time.time()
                
                updates.append({
                    "id": agent.id,
                    "output": agent.typed_text,
                    "score": agent.score,
                    "conversation_turn": turn_num + 1
                })
            
            # Batch broadcast updates for all agents
            if updates:
                await broadcast({
                    "type": "keystroke_update",
                    "turn": turn_num + 1,
                    "updates": updates,
                    "milestones": model_state.milestones.to_dict(),
                    "model_type": model_type.value
                })
            
            await asyncio.sleep(config.keystroke_interval)
        
        # Finalize sentences for Word Genome agents (add punctuation)
        if model_type == ModelType.WORD_GENOME:
            for agent in agents:
                if hasattr(agent, 'finalize_sentence') and agent.typed_text:
                    agent.finalize_sentence()
        
        # Store agent outputs
        turn_updates = []
        
        # For word-level models: Get GPT responses and set context for next turn
        if model_type in (ModelType.WORD_NN, ModelType.WORD_GENOME) and turn_num < config.word_nn_turns_per_gen - 1:
            # Get GPT responses in parallel
            gpt_tasks = []
            for agent in agents:
                conv_history = model_state.current_conversations.get(agent.id, [])
                gpt_tasks.append(get_gpt_response(agent.typed_text, conv_history))
            
            gpt_responses = await asyncio.gather(*gpt_tasks)
            
            for agent, gpt_response in zip(agents, gpt_responses):
                agent_text = agent.typed_text
                
                model_state.current_conversations[agent.id].append({
                    "agent_text": agent_text,
                    "openai_response": gpt_response
                })
                
                # Set GPT response as context for next turn
                if gpt_response:
                    agent.set_context(gpt_response)
                
                # Imitation learning: occasionally train on GPT patterns
                if gpt_response and random.random() < config.imitation_rate:
                    gpt_words = extract_words_from_response(gpt_response)
                    if len(gpt_words) >= 3:
                        agent.train_on_sequence(gpt_words)
                        print(f"[IMITATION] Agent {agent.id} learned from: '{gpt_response[:30]}...'")
                
                turn_updates.append({
                    "agent_id": agent.id,
                    "agent_text": agent_text,
                    "openai_response": gpt_response,
                    "turn": turn_num + 1
                })
        else:
            # No GPT response for last turn or non-Word NN models
            for agent in agents:
                agent_text = agent.typed_text
                
                model_state.current_conversations[agent.id].append({
                    "agent_text": agent_text,
                    "openai_response": ""
                })
                
                turn_updates.append({
                    "agent_id": agent.id,
                    "agent_text": agent_text,
                    "openai_response": "",
                    "turn": turn_num + 1
                })
        
        # Broadcast turn results
        await broadcast({
            "type": "turn_end",
            "turn": turn_num + 1,
            "updates": turn_updates,
            "model_type": model_type.value
        })
        
        await asyncio.sleep(0.2)  # Brief pause between turns
    
    if not model_state.is_running:
        return
    
    print(f"[DEBUG] Starting scoring for {len(agents)} agents...")
    
    # Score all conversations
    if model_type == ModelType.WORD_GENOME:
        # BATCH SCORING: One API call for all 12 agents
        sentences_to_score = []
        for agent in agents:
            conv_turns = model_state.current_conversations[agent.id]
            agent_text = conv_turns[0].get("agent_text", "") if conv_turns else ""
            sentences_to_score.append((agent.id, agent_text))
        
        gpt_context = model_state.gpt_first_message if model_state.current_phase == WordGenomePhase.CONVERSATION else None
        
        print(f"[WORD GENOME] Scoring {len(sentences_to_score)} sentences (Phase {model_state.current_phase.value})...")
        all_scores = await score_word_genome_openai(
            sentences_to_score, 
            phase=model_state.current_phase.value,
            gpt_context=gpt_context
        )
        print(f"[WORD GENOME] Scoring complete!")
    else:
        # Standard parallel scoring for other models
        scoring_tasks = []
        for agent in agents:
            conv_turns = model_state.current_conversations[agent.id]
            scoring_tasks.append(score_conversation(
                conv_turns, 
                agent.vocabulary,
                is_word_nn=(model_type == ModelType.WORD_NN)
            ))
        all_scores = await asyncio.gather(*scoring_tasks)
    
    print(f"[DEBUG] Scoring completed (Phase: {model_state.current_phase.name if model_type == ModelType.WORD_GENOME else 'N/A'})")
    
    agent_scores = {}
    patterns_discovered = False
    
    for agent, scores in zip(agents, all_scores):
        # Extract valid words found in this output
        words_found = scores.get("details", {}).get("valid_words", [])
        
        agent_scores[agent.id] = scores
        agent.set_score(scores["total"], scores)
        
        # Add new words to vocabulary
        if words_found:
            agent.add_words(words_found)
            # Also update global vocabulary
            state.global_vocabulary.update(w.lower() for w in words_found)
        
        # For Word Genome: Add all words from output to global vocabulary
        # This ensures 10k words used get added to "Words Found"
        if model_type == ModelType.WORD_GENOME:
            agent_text = model_state.current_conversations.get(agent.id, [{}])[0].get("agent_text", "")
            if agent_text:
                # Extract words and add to global vocabulary
                import re
                words_in_output = re.findall(r"[a-zA-Z]+", agent_text.lower())
                for word in words_in_output:
                    if len(word) >= 1:  # Include all words including 'a', 'i'
                        state.global_vocabulary.add(word)
        
        # Extract discovered patterns for word-level models
        if model_type in (ModelType.WORD_NN, ModelType.WORD_GENOME):
            # Extract bigrams
            new_bigrams = scores.get("discovered_bigrams", [])
            for bigram in new_bigrams:
                if bigram and bigram not in state.discovered_bigrams:
                    state.discovered_bigrams.add(bigram.lower())
                    patterns_discovered = True
                    print(f"[DISCOVERY] New bigram: '{bigram}'")
            
            # Extract trigrams
            new_trigrams = scores.get("discovered_trigrams", [])
            for trigram in new_trigrams:
                if trigram and trigram not in state.discovered_trigrams:
                    state.discovered_trigrams.add(trigram.lower())
                    patterns_discovered = True
                    print(f"[DISCOVERY] New trigram: '{trigram}'")
            
            # Check for valid sentence
            is_valid = scores.get("is_valid_sentence", False)
            coherence = scores.get("coherence_score", 0)
            agent_text = model_state.current_conversations.get(agent.id, [{}])[0].get("agent_text", "")
            
            if is_valid and coherence >= config.sentence_coherence_threshold and len(agent_text.split()) >= 3:
                # Check if sentence is not already saved (avoid duplicates)
                existing_texts = {s["text"] for s in state.discovered_sentences}
                if agent_text.strip().lower() not in existing_texts:
                    state.discovered_sentences.append({
                        "text": agent_text.strip(),
                        "score": scores["total"],
                        "coherence": coherence,
                        "agent_id": agent.id,
                        "generation": model_state.generation,
                        "model_type": model_type.value,
                        "timestamp": time.time()
                    })
                    # Keep only top sentences
                    state.discovered_sentences = sorted(
                        state.discovered_sentences,
                        key=lambda x: x["score"],
                        reverse=True
                    )[:config.max_discovered_sentences]
                    patterns_discovered = True
                    print(f"[DISCOVERY] New valid sentence: '{agent_text[:50]}...' (score: {scores['total']:.2f})")
        
        # Check milestones
        new_milestones = model_state.milestones.check(scores)
        if new_milestones:
            await broadcast({
                "type": "milestone",
                "milestones": new_milestones,
                "agent_id": agent.id,
                "details": scores,
                "model_type": model_type.value
            })
    
    # Save patterns to disk if any were discovered
    if patterns_discovered:
        save_patterns(state)
    
    # Get stats
    scores_list = [a.score for a in agents]
    stats = {
        "generation": model_state.generation,
        "mean": sum(scores_list) / len(scores_list) if scores_list else 0,
        "max": max(scores_list) if scores_list else 0,
        "min": min(scores_list) if scores_list else 0,
        "best_score": max(scores_list) if scores_list else 0,
        "exploration_rate": agents[0].exploration_rate if agents else 0
    }
    model_state.history.append(stats)
    
    # Track best outputs
    best_agent = max(agents, key=lambda a: a.score)
    best_conv = model_state.current_conversations.get(best_agent.id, [])
    
    if best_agent.score > 0.1:
        model_state.best_outputs.append({
            "generation": model_state.generation,
            "conversation": best_conv,
            "score": best_agent.score,
            "scores_breakdown": agent_scores.get(best_agent.id, {})
        })
        model_state.best_outputs = sorted(
            model_state.best_outputs, 
            key=lambda x: x["score"], 
            reverse=True
        )[:20]
    
    # Save interesting conversations and best genome to disk
    interesting_to_save = []
    for agent in agents:
        conv = model_state.current_conversations.get(agent.id, [])
        scores = agent_scores.get(agent.id, {})
        if is_interesting_conversation(conv, scores):
            interesting_to_save.append({
                "generation": model_state.generation,
                "agent_id": agent.id,
                "conversation": conv,
                "scores": scores,
                "timestamp": time.time()
            })
    
    if interesting_to_save:
        save_to_json(INTERESTING_CONVS_FILE, interesting_to_save)
    
    # Save best agent's metadata (model weights saved separately via save_all_agents)
    save_to_json(BEST_AGENTS_FILE, [{
        "generation": model_state.generation,
        "agent_id": best_agent.id,
        "score": best_agent.score,
        "exploration_rate": best_agent.exploration_rate,
        "vocabulary_size": len(best_agent.vocabulary),
        "model_type": model_type.value,
        "timestamp": time.time()
    }])
    
    # Broadcast generation results with discovered patterns
    await broadcast({
        "type": "generation_end",
        "generation": model_state.generation,
        "stats": stats,
        "agents": [a.to_dict() for a in agents],
        "best_outputs": model_state.best_outputs[:5],
        "milestones": model_state.milestones.to_dict(),
        "conversations": {
            aid: conv for aid, conv in model_state.current_conversations.items()
        },
        "global_vocabulary": list(state.global_vocabulary),
        "model_type": model_type.value,
        # Pattern discoveries
        "discovered_bigrams": list(state.discovered_bigrams),
        "discovered_trigrams": list(state.discovered_trigrams),
        "discovered_sentences": state.discovered_sentences[:20]  # Top 20
    })
    
    # Pause to show scores before next generation
    await asyncio.sleep(1.5)
    
    # Model-specific learning updates
    avg_score = stats.get("mean", 0.0)
    
    if model_type == ModelType.GENOME:
        # Genetic algorithm evolution
        model_state.agents = evolve_generation(agents, state.global_vocabulary)
        for agent in model_state.agents:
            agent.decay_exploration()
    
    elif model_type == ModelType.CHAR_NN:
        # Neural network learning: REINFORCE
        alpha = 0.1
        model_state.baseline_reward = alpha * avg_score + (1 - alpha) * model_state.baseline_reward
        
        for agent in agents:
            if agent.score > 0:
                agent.update(reward=agent.score, baseline=model_state.baseline_reward)
            agent.decay_exploration()
        
        save_all_agents(agents)
    
    elif model_type == ModelType.WORD_NN:
        # Word-level neural network learning: REINFORCE
        print(f"[DEBUG] Word NN learning phase starting...")
        alpha = 0.1
        model_state.baseline_reward = alpha * avg_score + (1 - alpha) * model_state.baseline_reward
        
        for agent in agents:
            # Update vocabulary with global vocab
            agent.update_vocabulary(state.global_vocabulary)
            
            if agent.score > 0:
                agent.update(reward=agent.score, baseline=model_state.baseline_reward)
            agent.decay_exploration()
        
        print(f"[DEBUG] Saving word agents...")
        save_word_agents(agents)
        print(f"[DEBUG] Word agents saved")
    
    elif model_type == ModelType.WORD_GENOME:
        # Record phase metrics for curriculum learning
        agent_results = [agent_scores.get(a.id, {}) for a in agents]
        record_phase_metrics(model_state, agent_results)
        
        # Learn from successful agents with structure bonus
        sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
        for agent in sorted_agents[:3]:
            structure_score = agent_scores.get(agent.id, {}).get("structure_score", 0.0)
            learn_from_success_word_genome(agent, structure_score)
        
        # Word-level genetic algorithm evolution (pass generation for lineage tracking)
        model_state.agents = evolve_word_genome_generation(agents, model_state.generation)
        for agent in model_state.agents:
            agent.decay_exploration()
        
        # Check for phase transition
        phase_before = model_state.current_phase
        transitioned = transition_phase(model_state)
        if transitioned:
            await broadcast({
                "type": "phase_transition",
                "old_phase": phase_before.name,
                "new_phase": model_state.current_phase.name,
                "generation": model_state.generation
            })
        
        # Save best Word Genome agents (every 5 generations to reduce I/O)
        if model_state.generation % 5 == 0:
            save_word_genome_agents(model_state.agents)
    
    model_state.generation += 1
    print(f"[DEBUG] Generation {model_state.generation} completed for {model_type}", flush=True)


async def run_concurrent_generation():
    """Run Char NN and Word NN models simultaneously."""
    char_state = state.model_states[ModelType.CHAR_NN]
    word_state = state.model_states[ModelType.WORD_NN]
    
    # Broadcast generation start for concurrent mode
    await broadcast({
        "type": "generation_start",
        "mode": "concurrent",
        "char_nn": {
            "generation": char_state.generation,
            "agents": [a.to_dict() for a in char_state.agents[:6]]  # 6 agents each
        },
        "word_nn": {
            "generation": word_state.generation,
            "agents": [a.to_dict() for a in word_state.agents[:6]]
        }
    })
    
    # Reset agents
    for agent in char_state.agents[:6]:
        agent.reset()
    for agent in word_state.agents[:6]:
        agent.reset()
    
    # Initialize conversation tracking
    char_state.current_conversations = {a.id: [] for a in char_state.agents[:6]}
    word_state.current_conversations = {a.id: [] for a in word_state.agents[:6]}
    
    # Run generation loop
    start_time = time.time()
    char_last_output = {a.id: start_time for a in char_state.agents[:6]}
    word_last_output = {a.id: start_time for a in word_state.agents[:6]}
    char_finished = set()
    word_finished = set()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= config.generation_time:
            break
        
        # Check if all agents finished
        if len(char_finished) >= 6 and len(word_finished) >= 6:
            break
        
        char_updates = []
        word_updates = []
        
        # Process Char NN agents
        for agent in char_state.agents[:6]:
            if agent.id in char_finished:
                continue
            
            if time.time() - char_last_output[agent.id] >= config.idle_timeout and agent.typed_text:
                char_finished.add(agent.id)
                continue
            
            if detect_sentence_end(agent.typed_text) and len(agent.typed_text) > 5:
                char_finished.add(agent.id)
                continue
            
            char = agent.step()
            if char:
                char_last_output[agent.id] = time.time()
            
            char_updates.append({
                "id": agent.id,
                "output": agent.typed_text,
                "score": agent.score
            })
        
        # Process Word NN agents
        for agent in word_state.agents[:6]:
            if agent.id in word_finished:
                continue
            
            if time.time() - word_last_output[agent.id] >= config.idle_timeout and agent.typed_text:
                word_finished.add(agent.id)
                continue
            
            # Word count limit for Word NN
            word_count = len(agent.typed_text.split())
            if word_count >= config.word_nn_max_length:
                word_finished.add(agent.id)
                continue
            
            word = agent.step()
            if word:
                word_last_output[agent.id] = time.time()
            
            word_updates.append({
                "id": agent.id,
                "output": agent.typed_text,
                "score": agent.score
            })
        
        # Broadcast concurrent updates
        if char_updates or word_updates:
            await broadcast({
                "type": "keystroke_update",
                "mode": "concurrent",
                "char_nn": {"updates": char_updates},
                "word_nn": {"updates": word_updates}
            })
        
        await asyncio.sleep(config.keystroke_interval)
    
    # Score agents and update states
    char_scores = []
    word_scores = []
    
    for agent in char_state.agents[:6]:
        agent_text = agent.typed_text
        char_state.current_conversations[agent.id].append({
            "agent_text": agent_text,
            "openai_response": ""
        })
        
        scores = await score_conversation(
            [{"agent_text": agent_text, "openai_response": ""}],
            agent_vocab=agent.vocabulary
        )
        agent.set_score(scores["total"], scores.get("details", {}))
        char_scores.append(agent.score)
        
        # Update global vocabulary with found words
        words_found = scores.get("details", {}).get("valid_words", [])
        if words_found:
            agent.add_words(words_found)
            state.global_vocabulary.update(words_found)
    
    for agent in word_state.agents[:6]:
        agent_text = agent.typed_text
        word_state.current_conversations[agent.id].append({
            "agent_text": agent_text,
            "openai_response": ""
        })
        
        scores = await score_conversation(
            [{"agent_text": agent_text, "openai_response": ""}],
            agent_vocab=agent.vocabulary
        )
        agent.set_score(scores["total"], scores.get("details", {}))
        word_scores.append(agent.score)
    
    # Calculate stats for both models
    char_stats = {
        "mean": sum(char_scores) / len(char_scores) if char_scores else 0,
        "max": max(char_scores) if char_scores else 0,
        "min": min(char_scores) if char_scores else 0,
        "best_score": max(char_scores) if char_scores else 0
    }
    word_stats = {
        "mean": sum(word_scores) / len(word_scores) if word_scores else 0,
        "max": max(word_scores) if word_scores else 0,
        "min": min(word_scores) if word_scores else 0,
        "best_score": max(word_scores) if word_scores else 0
    }
    
    # Update history
    char_state.history.append({"generation": char_state.generation, **char_stats})
    word_state.history.append({"generation": word_state.generation, **word_stats})
    
    # Broadcast generation end
    await broadcast({
        "type": "generation_end",
        "mode": "concurrent",
        "char_nn": {
            "generation": char_state.generation,
            "stats": char_stats,
            "agents": [a.to_dict() for a in char_state.agents[:6]],
            "history": char_state.history[-50:]
        },
        "word_nn": {
            "generation": word_state.generation,
            "stats": word_stats,
            "agents": [a.to_dict() for a in word_state.agents[:6]],
            "history": word_state.history[-50:]
        },
        "global_vocabulary": list(state.global_vocabulary)
    })
    
    # Apply learning updates for both models
    # Char NN - REINFORCE
    char_avg = char_stats["mean"]
    char_state.baseline_reward = 0.1 * char_avg + 0.9 * char_state.baseline_reward
    for agent in char_state.agents[:6]:
        if agent.score > 0:
            agent.update(reward=agent.score, baseline=char_state.baseline_reward)
        agent.decay_exploration()
    save_all_agents(char_state.agents)
    
    # Word NN - REINFORCE
    word_avg = word_stats["mean"]
    word_state.baseline_reward = 0.1 * word_avg + 0.9 * word_state.baseline_reward
    for agent in word_state.agents[:6]:
        agent.update_vocabulary(state.global_vocabulary)
        if agent.score > 0:
            agent.update(reward=agent.score, baseline=word_state.baseline_reward)
        agent.decay_exploration()
    save_word_agents(word_state.agents)
    
    # Increment generations
    char_state.generation += 1
    word_state.generation += 1


async def model_training_loop(model_type: ModelType):
    """Training loop for a specific model type - runs until that model is stopped."""
    model_state = state.model_states[model_type]
    print(f"[DEBUG] Training loop started for model: {model_type}")
    
    while model_state.is_running:
        try:
            await run_conversation_generation(model_type)
        except Exception as e:
            print(f"[ERROR] Generation error for {model_type}: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(0.1)  # Minimal pause between generations
    
    print(f"[DEBUG] Training loop ended for model: {model_type}")


async def training_loop():
    """Legacy training loop - for backward compatibility.
    
    Starts the current model's training loop.
    """
    model_type = state.current_model
    
    if model_type == ModelType.CONCURRENT:
        # For concurrent mode, start both Char NN and Word NN
        await start_model_training(ModelType.CHAR_NN)
        await start_model_training(ModelType.WORD_NN)
    else:
        await start_model_training(model_type)


async def start_model_training(model_type: ModelType):
    """Start training for a specific model."""
    if model_type == ModelType.CONCURRENT:
        # Start both Char NN and Word NN
        await start_model_training(ModelType.CHAR_NN)
        await start_model_training(ModelType.WORD_NN)
        return
    
    model_state = state.model_states[model_type]
    
    if model_state.is_running:
        print(f"[DEBUG] {model_type} already running")
        return
    
    model_state.is_running = True
    model_state.task = asyncio.create_task(model_training_loop(model_type))
    print(f"[DEBUG] Started training for {model_type}")


async def stop_model_training(model_type: ModelType):
    """Stop training for a specific model."""
    if model_type == ModelType.CONCURRENT:
        # Stop both Char NN and Word NN
        await stop_model_training(ModelType.CHAR_NN)
        await stop_model_training(ModelType.WORD_NN)
        return
    
    model_state = state.model_states[model_type]
    model_state.is_running = False
    
    if model_state.task:
        model_state.task.cancel()
        try:
            await model_state.task
        except asyncio.CancelledError:
            pass
        model_state.task = None
    
    print(f"[DEBUG] Stopped training for {model_type}")


@app.post("/start")
async def start_training():
    """Start or resume training for the current model."""
    model_type = state.current_model
    
    # Check if agents exist
    if model_type == ModelType.CONCURRENT:
        char_agents = state.model_states[ModelType.CHAR_NN].agents
        word_agents = state.model_states[ModelType.WORD_NN].agents
        if not char_agents or not word_agents:
            state.initialize()
    else:
        if not state.model_states[model_type].agents:
            state.initialize()
    
    await start_model_training(model_type)
    
    gen = state.generation if model_type != ModelType.CONCURRENT else -1
    return {"status": "started", "model": model_type.value, "generation": gen}


@app.post("/stop")
async def stop_training():
    """Pause training for the current model."""
    model_type = state.current_model
    await stop_model_training(model_type)
    return {"status": "stopped", "model": model_type.value, "generation": state.generation}


@app.post("/start/{model_type}")
async def start_specific_model(model_type: str):
    """Start training for a specific model."""
    try:
        mt = ModelType(model_type)
    except ValueError:
        return {"error": f"Invalid model type: {model_type}"}
    
    # Check if agents exist
    if mt == ModelType.CONCURRENT:
        char_agents = state.model_states[ModelType.CHAR_NN].agents
        word_agents = state.model_states[ModelType.WORD_NN].agents
        if not char_agents or not word_agents:
            state.initialize()
    else:
        if not state.model_states[mt].agents:
            state.initialize()
    
    await start_model_training(mt)
    
    gen = state.model_states[mt].generation if mt != ModelType.CONCURRENT else -1
    return {"status": "started", "model": mt.value, "generation": gen}


@app.post("/stop/{model_type}")
async def stop_specific_model(model_type: str):
    """Stop training for a specific model."""
    try:
        mt = ModelType(model_type)
    except ValueError:
        return {"error": f"Invalid model type: {model_type}"}
    
    await stop_model_training(mt)
    gen = state.model_states[mt].generation if mt != ModelType.CONCURRENT else -1
    return {"status": "stopped", "model": mt.value, "generation": gen}


class ResetRequest(BaseModel):
    keep_agents: bool = False


@app.post("/reset")
async def reset_training(request: ResetRequest = ResetRequest()):
    """Reset training to initial state.
    
    Args:
        keep_agents: If True, keeps trained neural networks. If False, creates fresh ones.
    """
    await stop_training()
    state.reset(keep_agents=request.keep_agents)
    
    await broadcast({
        "type": "reset",
        "agents": [a.to_dict() for a in state.agents],
        "milestones": state.milestones.to_dict(),
        "global_vocabulary": list(state.global_vocabulary),
        "current_model": state.current_model.value
    })
    
    return {"status": "reset", "kept_agents": request.keep_agents}


class SwitchModelRequest(BaseModel):
    model: str  # "genome", "char_nn", "word_nn", or "concurrent"


@app.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    """Switch to a different model VIEW (does not stop running models).
    
    Each model maintains its own state (generation, agents, history).
    Global vocabulary is shared across all models.
    'concurrent' runs Char NN and Word NN side-by-side.
    
    Note: This only changes the view, it does NOT stop training for any model.
    """
    try:
        new_model = ModelType(request.model)
    except ValueError:
        return {"error": f"Invalid model type: {request.model}. Use 'genome', 'char_nn', 'word_nn', or 'concurrent'"}
    
    # Switch model
    state.switch_model(new_model)
    
    # Broadcast the switch - different format for concurrent mode
    # Include running status for all models
    running_status = {
        "genome": state.model_states[ModelType.GENOME].is_running,
        "char_nn": state.model_states[ModelType.CHAR_NN].is_running,
        "word_nn": state.model_states[ModelType.WORD_NN].is_running
    }
    
    if new_model == ModelType.CONCURRENT:
        char_state = state.model_states[ModelType.CHAR_NN]
        word_state = state.model_states[ModelType.WORD_NN]
        await broadcast({
            "type": "model_switch",
            "mode": "concurrent",
            "current_model": state.current_model.value,
            "global_vocabulary": list(state.global_vocabulary),
            "running_status": running_status,
            "char_nn": {
                "generation": char_state.generation,
                "agents": [a.to_dict() for a in char_state.agents[:6]],
                "history": char_state.history[-50:],
                "best_outputs": char_state.best_outputs[:5],
                "milestones": char_state.milestones.to_dict(),
                "is_running": char_state.is_running
            },
            "word_nn": {
                "generation": word_state.generation,
                "agents": [a.to_dict() for a in word_state.agents[:6]],
                "history": word_state.history[-50:],
                "best_outputs": word_state.best_outputs[:5],
                "milestones": word_state.milestones.to_dict(),
                "is_running": word_state.is_running
            }
        })
    else:
        model_state = state.model_states[new_model]
        await broadcast({
            "type": "model_switch",
            "current_model": state.current_model.value,
            "generation": model_state.generation,
            "agents": [a.to_dict() for a in model_state.agents],
            "history": model_state.history[-50:],
            "best_outputs": model_state.best_outputs[:10],
            "milestones": model_state.milestones.to_dict(),
            "global_vocabulary": list(state.global_vocabulary),
            "running_status": running_status,
            "is_running": model_state.is_running
        })
    
    return {
        "status": "switched",
        "current_model": state.current_model.value,
        "generation": state.generation if new_model != ModelType.CONCURRENT else -1,
        "num_agents": len(state.agents) if new_model != ModelType.CONCURRENT else 12
    }


@app.get("/models")
async def get_models():
    """Get info about all available models."""
    models_info = {}
    for model_type, model_state in state.model_states.items():
        models_info[model_type.value] = {
            "generation": model_state.generation,
            "num_agents": len(model_state.agents),
            "best_score": max([a.score for a in model_state.agents], default=0) if model_state.agents else 0,
            "is_running": model_state.is_running
        }
    
    return {
        "current_model": state.current_model.value,
        "models": models_info,
        "global_vocabulary_size": len(state.global_vocabulary)
    }


@app.get("/state")
async def get_state():
    """Get current training state."""
    return {
        "is_running": state.is_running,
        "generation": state.generation,
        "num_agents": len(state.agents),
        "agents": [a.to_dict() for a in state.agents],
        "history": state.history[-50:],
        "best_outputs": state.best_outputs[:10],
        "memory": pattern_memory.to_dict(),
        "milestones": state.milestones.to_dict(),
        "current_model": state.current_model.value,
        "global_vocabulary": list(state.global_vocabulary),
        "config": {
            "num_agents": config.num_agents,
            "generation_time": config.generation_time,
            "turns_per_conversation": config.turns_per_conversation,
            "mutation_rate": config.mutation_rate,
            "exploration_decay": config.exploration_decay
        }
    }


@app.post("/config")
async def update_config(update: ConfigUpdate):
    """Update configuration parameters."""
    if update.num_agents is not None:
        config.num_agents = update.num_agents
    if update.generation_time is not None:
        config.generation_time = update.generation_time
    if update.mutation_rate is not None:
        config.mutation_rate = update.mutation_rate
    
    return {"status": "updated", "config": {
        "num_agents": config.num_agents,
        "generation_time": config.generation_time,
        "mutation_rate": config.mutation_rate
    }}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    state.connected_clients.append(websocket)
    
    try:
        # Send initial state with running status for all models
        running_status = {
            "genome": state.model_states[ModelType.GENOME].is_running,
            "char_nn": state.model_states[ModelType.CHAR_NN].is_running,
            "word_nn": state.model_states[ModelType.WORD_NN].is_running
        }
        
        if state.current_model == ModelType.CONCURRENT:
            char_state = state.model_states[ModelType.CHAR_NN]
            word_state = state.model_states[ModelType.WORD_NN]
            init_msg = {
                "type": "init",
                "mode": "concurrent",
                "is_running": state.is_running,
                "running_status": running_status,
                "total_turns": config.turns_per_conversation,
                "global_vocabulary": list(state.global_vocabulary),
                "current_model": state.current_model.value,
                # Pattern discoveries
                "discovered_bigrams": list(state.discovered_bigrams),
                "discovered_trigrams": list(state.discovered_trigrams),
                "discovered_sentences": state.discovered_sentences[:20],
                "char_nn": {
                    "generation": char_state.generation,
                    "agents": [a.to_dict() for a in char_state.agents[:6]],
                    "history": char_state.history[-50:],
                    "best_outputs": char_state.best_outputs[:5],
                    "milestones": char_state.milestones.to_dict(),
                    "is_running": char_state.is_running
                },
                "word_nn": {
                    "generation": word_state.generation,
                    "agents": [a.to_dict() for a in word_state.agents[:6]],
                    "history": word_state.history[-50:],
                    "best_outputs": word_state.best_outputs[:5],
                    "milestones": word_state.milestones.to_dict(),
                    "is_running": word_state.is_running
                }
            }
        else:
            model_state = state.model_states[state.current_model]
            init_msg = {
                "type": "init",
                "generation": model_state.generation,
                "is_running": model_state.is_running,
                "running_status": running_status,
                "agents": [a.to_dict() for a in model_state.agents],
                "history": model_state.history[-50:],
                "best_outputs": model_state.best_outputs[:10],
                "milestones": model_state.milestones.to_dict(),
                "total_turns": config.turns_per_conversation,
                "global_vocabulary": list(state.global_vocabulary),
                "current_model": state.current_model.value,
                # Pattern discoveries
                "discovered_bigrams": list(state.discovered_bigrams),
                "discovered_trigrams": list(state.discovered_trigrams),
                "discovered_sentences": state.discovered_sentences[:20],
                # Word Genome phase tracking
                "word_genome_phase": model_state.current_phase.name if state.current_model == ModelType.WORD_GENOME else None,
                "word_genome_phase_progress": model_state.phase_progress if state.current_model == ModelType.WORD_GENOME else 0.0
            }
        await websocket.send_text(json.dumps(init_msg))
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                msg = json.loads(data)
                msg_type = msg.get("type")
                print(f"[WS] Received message type: {msg_type}", flush=True)  # Debug logging
                
                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg_type == "set_use_10k":
                    state.use_10k_words = msg.get("use_10k", True)
                    print(f"[WS] use_10k_words set to: {state.use_10k_words}", flush=True)
                    
                    # Compute vocab for non-10k mode
                    custom_vocab = None if state.use_10k_words else state.global_vocabulary
                    print(f"[WS] Custom vocab size: {len(custom_vocab) if custom_vocab else 'N/A (using 10k)'}")
                    
                    # Re-initialize Word NN agents with new vocab setting
                    # IMPORTANT: load_existing=False to create fresh agents
                    state.model_states[ModelType.WORD_NN].agents = create_word_agent_pool(
                        config.num_agents,
                        shared_vocabulary=custom_vocab,
                        load_existing=False,
                        use_10k=state.use_10k_words
                    )
                    
                    # Re-initialize Word Genome agents with new vocab setting
                    # IMPORTANT: load_existing=False to create fresh agents
                    state.model_states[ModelType.WORD_GENOME].agents = create_word_genome_agent_pool(
                        config.num_agents,
                        load_existing=False,  # Force new agents with new vocab
                        custom_vocab=custom_vocab
                    )
                    print(f"[WS] Re-initialized Word NN and Word Genome agents with use_10k={state.use_10k_words}")
                    
                    # Broadcast updated state with new agents
                    await broadcast_state()
                elif msg_type == "reset_vocab":
                    state.global_vocabulary.clear()
                    state.discovered_bigrams.clear()
                    state.discovered_trigrams.clear()
                    state.discovered_sentences.clear()
                    
                    # Clear patterns file
                    patterns_file = os.path.join(os.path.dirname(__file__), "data", "discovered_patterns.json")
                    if os.path.exists(patterns_file):
                        os.remove(patterns_file)
                    
                    print(f"[WS] Vocabulary and patterns reset!")
                    # Broadcast updated state
                    await websocket.send_text(json.dumps({
                        "type": "vocab_reset",
                        "global_vocabulary": [],
                        "discovered_bigrams": [],
                        "discovered_trigrams": [],
                        "discovered_sentences": []
                    }))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)


@app.get("/saved-conversations")
async def get_saved_conversations():
    """Get saved interesting conversations."""
    return {
        "conversations": load_from_json(INTERESTING_CONVS_FILE)[-100:]  # Last 100
    }


@app.get("/best-agents")
async def get_best_agents():
    """Get saved best agents."""
    agents = load_from_json(BEST_AGENTS_FILE)
    return {
        "agents": sorted(agents, key=lambda x: x.get("score", 0), reverse=True)[:50]
    }


class ChatRequest(BaseModel):
    message: str
    use_best_genome: bool = True


class ScoreTextRequest(BaseModel):
    text: str
    context: str = ""


@app.post("/score-text")
async def score_text(request: ScoreTextRequest):
    """Score arbitrary text using the OpenAI judge.
    
    This allows testing what scores a proper English sentence would get.
    Uses pure OpenAI scoring without heuristic blending for fair baseline comparison.
    """
    from scorer import score_with_openai
    
    if not request.text or len(request.text.strip()) < 2:
        return {
            "error": "Text too short",
            "scores": None
        }
    
    try:
        # Use is_word_nn=True for pure OpenAI scoring without heuristic blending
        # This gives a fair baseline for comparing human-written text
        scores = await score_with_openai(
            request.text, 
            conversation_context=request.context if request.context else None,
            is_word_nn=True  # Pure OpenAI scoring
        )
        return {
            "text": request.text,
            "scores": {
                "vocabulary": scores["vocabulary_score"],
                "grammar": scores["grammar_score"],
                "coherence": scores["coherence_score"],
                "total": scores["total"],
                "reason": scores.get("reason", ""),
                "valid_words": scores.get("valid_words", [])
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "scores": None
        }


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """Chat with the best trained Word NN agent.
    
    Uses the Word NN model for coherent word-level generation.
    """
    best_agent = None
    
    # Use Word NN agents only
    word_agents = state.model_states[ModelType.WORD_NN].agents
    if word_agents:
        best_agent = max(word_agents, key=lambda a: a.score)
    
    if best_agent is None or best_agent.score <= 0:
        return {
            "response": "No trained Word NN agents yet. Start Word NN training first!",
            "agent_id": None
        }
    
    # Generate response using Word NN (word-level generation)
    best_agent.reset()
    best_agent.exploration_rate = 0.05  # Low exploration for inference
    
    words = []
    for _ in range(20):  # Max 20 words
        word = best_agent.step()
        if word:
            words.append(word)
        # Stop on sentence-ending punctuation
        if best_agent.typed_text and best_agent.typed_text.endswith(('.', '!', '?')):
            break
    
    response = best_agent.typed_text or " ".join(words) or "(empty response)"
    
    return {
        "response": response,
        "agent_id": best_agent.id,
        "agent_score": best_agent.score,
        "agent_type": "word_nn",
        "vocabulary_size": len(best_agent.vocabulary) if hasattr(best_agent, 'vocabulary') else 0
    }


@app.get("/agent/{agent_id}/viz")
async def get_agent_visualization(agent_id: str):
    """Get neural network visualization data for a specific agent."""
    # Search all model types for the agent
    agent = None
    agent_type = None
    
    for model_type, model_state in state.model_states.items():
        for a in model_state.agents:
            if a.id == agent_id:
                agent = a
                agent_type = model_type
                break
        if agent:
            break
    
    if not agent:
        return {"error": "Agent not found", "viz": None}
    
    try:
        viz_data = agent.get_visualization_data()
        
        # Normalize data structure for frontend
        # Word NN returns word_probs/context_words, convert to char_probs/context_chars format
        if agent_type == ModelType.WORD_NN and viz_data:
            # Convert word_probs to char_probs format (frontend expects this)
            if "word_probs" in viz_data and "char_probs" not in viz_data:
                viz_data["char_probs"] = [
                    {"char": wp["word"], "prob": wp["prob"]} 
                    for wp in viz_data.get("word_probs", [])
                ]
            if "context_words" in viz_data and "context_chars" not in viz_data:
                viz_data["context_chars"] = viz_data.get("context_words", [])
            viz_data["is_word_nn"] = True
        else:
            viz_data["is_word_nn"] = False
        
        return {
            "agent_id": agent_id,
            "typed_text": agent.typed_text,
            "agent_type": agent_type.value if agent_type else None,
            "viz": viz_data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "viz": None}


if __name__ == "__main__":
    import uvicorn
    # Use PORT env var for Railway, default to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))
    print(f"[SERVER] Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
