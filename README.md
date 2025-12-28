# rl.english

A multi-agent reinforcement learning system where AI agents learn English from scratch through evolutionary selection and neural networks. Watch agents progress from random characters to coherent sentences.

## Model Types

| Model | Description |
|-------|-------------|
| **Genome** | Genetic algorithm with character-level probability distributions |
| **Char NN** | Character-level transformer neural network |
| **Word NN** | Word-level transformer using discovered vocabulary |
| **Word Genome** | Word-level genetic algorithm with curriculum learning |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (port 3000)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Agent Grid  │  │Score Graph  │  │   Best Outputs      │  │
│  │ (real-time) │  │ (history)   │  │   (hall of fame)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Pattern Banks: Words | Bigrams | Trigrams | Sentences│   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket
┌──────────────────────────┴──────────────────────────────────┐
│                   FastAPI Backend (port 8000)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Agents    │  │  Evolution  │  │  OpenAI Scorer      │  │
│  │ (4 models)  │  │  + NN Train │  │  (phase rubrics)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Set up the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Start the server
python main.py
```

### 2. Set up the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

### 3. Open the UI

Navigate to http://localhost:3000 and click **Start** to begin training.

## Word Genome Curriculum Learning

The Word Genome model uses a 4-phase curriculum:

| Phase | Focus | Criteria |
|-------|-------|----------|
| **1. Structure** | Capitals, punctuation, spacing | 80%+ proper formatting |
| **2. Patterns** | Bigrams, word combinations | Consistent valid bigrams |
| **3. Sentences** | SVO structure, coherence | Complete meaningful sentences |
| **4. Conversation** | GPT dialogue, response relevance | Back-and-forth conversation |

## Scoring System

- **Phase 1-2**: Local structure + GPT-4 pattern scoring
- **Phase 3-4**: Pure GPT-4 scoring with phase-specific rubrics
- Each agent receives individual feedback and reasoning

## Configuration

Edit `backend/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_agents | 12 | Agents per generation |
| mutation_rate | 0.1 | Probability of genome mutation |
| elitism_ratio | 0.2 | Top % kept unchanged |
| initial_exploration | 0.5 | Starting random action probability |
| exploration_decay | 0.99 | Per-generation exploration reduction |

## API Endpoints

- `POST /start` - Start training
- `POST /stop` - Pause training
- `POST /reset` - Reset to initial state
- `GET /state` - Get current state
- `POST /switch_model` - Switch between model types
- `WS /ws` - WebSocket for real-time updates

## Deployment

This project is configured for Vercel deployment:

```bash
# Push to GitHub
git add .
git commit -m "Initial commit"
git push origin main

# Connect to Vercel and deploy
```

Note: The backend requires a separate hosting solution (Railway, Render, etc.) with WebSocket support.

## Requirements

- Python 3.11+
- Node.js 18+
- OpenAI API key (GPT-4o-mini for scoring)

## License

MIT
