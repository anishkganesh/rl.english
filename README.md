# rl.english

A multi-agent reinforcement learning system where AI agents learn English from scratch through evolutionary selection and neural networks. Watch agents progress from random characters to coherent sentences.

## Model Types

| Model | Description |
|-------|-------------|
| **Genome** | Genetic algorithm with character-level probability distributions |
| **Char NN** | Character-level transformer neural network |
| **Word NN** | Word-level transformer using discovered vocabulary |
| **Word Genome** | Word-level genetic algorithm with curriculum learning |

## Quick Start (Local Development)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Start the server
python main.py
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

### 3. Open http://localhost:3000 and click **Start**

---

## 🚀 Production Deployment

### Architecture

```
┌─────────────────────────────────────────┐
│           Vercel (Frontend)             │
│         Static React/Vite App           │
│   Connects via WebSocket to Backend     │
└────────────────────┬────────────────────┘
                     │ wss://
┌────────────────────▼────────────────────┐
│          Railway (Backend)              │
│       Python FastAPI + WebSocket        │
│         + OpenAI API calls              │
└─────────────────────────────────────────┘
```

### Step 1: Deploy Backend to Railway

1. **Go to [Railway.app](https://railway.app)** and sign in with GitHub

2. **Create New Project** → **Deploy from GitHub repo**

3. **Select your `rl.english` repository**

4. **Configure the service:**
   - Root Directory: `backend`
   - Start Command: `python main.py`

5. **Add Environment Variable:**
   - Click on the service → **Variables** tab
   - Add: `OPENAI_API_KEY` = `sk-your-openai-key-here`

6. **Generate Domain:**
   - Go to **Settings** → **Networking** → **Generate Domain**
   - Copy the domain (e.g., `rl-english-production.up.railway.app`)

7. **Wait for deployment** (2-3 minutes)

### Step 2: Deploy Frontend to Vercel

1. **Go to [Vercel.com](https://vercel.com)** and sign in with GitHub

2. **Add New Project** → **Import** your `rl.english` repository

3. **Configure Project:**
   - Framework Preset: **Vite**
   - Root Directory: `./` (leave as default)
   - Build Command: `cd frontend && npm install && npm run build`
   - Output Directory: `frontend/dist`

4. **Add Environment Variable:**
   - Expand **Environment Variables**
   - Add: `VITE_BACKEND_URL` = `your-railway-domain.up.railway.app`
   - ⚠️ **Do NOT include `https://` or `wss://`** - just the domain

5. **Click Deploy** and wait (1-2 minutes)

6. **Done!** Your app is live at `your-project.vercel.app`

---

## Environment Variables Summary

### Backend (Railway)
| Variable | Value | Required |
|----------|-------|----------|
| `OPENAI_API_KEY` | `sk-your-key-here` | Yes |

### Frontend (Vercel)
| Variable | Value | Required |
|----------|-------|----------|
| `VITE_BACKEND_URL` | `your-app.railway.app` | Yes |

---

## Word Genome Curriculum Learning

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

## Requirements

- Python 3.11+
- Node.js 18+
- OpenAI API key (GPT-4o-mini for scoring)

## License

MIT
