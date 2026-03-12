# StudyLens Web App — Process Flow

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend   │────▶│  FastAPI Backend  │────▶│  Gemini 2.0     │
│  (Browser)   │◀────│  (app/main.py)   │◀────│  Flash API      │
└─────────────┘     └────────┬─────────┘     └─────────────────┘
                             │
                    ┌────────┴─────────┐
                    │  MiniLM Embedder │
                    │  (all-MiniLM-    │
                    │   L6-v2)         │
                    └──────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend framework | **FastAPI** | Async, auto-generated /docs, Pydantic validation |
| LLM (summarization + chat) | **Google Gemini 2.0 Flash** | Free tier (15 RPM, 1M tokens/day), 1M context window |
| Embedding model | **all-MiniLM-L6-v2** | 384-dim vectors, fast, fits in CPU memory |
| Similarity search | **scikit-learn cosine_similarity** | No external vector DB needed for 10 topics |
| Environment config | **python-dotenv** | Keeps API keys out of code |

## Data Flow

### Source Data

All lecture content lives in `data/processed/` as `{topic_key}_ori.txt` files.
Each file is a combination of three sources produced by the offline pipeline
(`scripts/make_dataset.py` + `scripts/build_features.py`):

```
{topic}_ori.txt = PPTX slide text + denoised transcript + student notes
```

The web app **reads** these files but never modifies them.

## Endpoint Process Flows

### 1. Topic Summary Generation

```
User clicks topic in sidebar
        │
        ▼
GET /api/topic/{key}/summary
        │
        ├── Cache hit? ──▶ Return cached summary instantly
        │
        ├── Read source text from data/processed/{key}_ori.txt
        │
        ├── Send to Gemini with structured summarization prompt
        │   (requesting title, key concepts, definitions, 300-500 words)
        │
        ├── Cache the result in memory
        │
        └── Return { summary, word_count, model: "Gemini 2.0 Flash" }
```

- **?regenerate=true** bypasses cache to produce a fresh summary
- Summaries are cached in `_summary_cache` (dict) for the server lifetime

### 2. RAG Chatbot (Q&A)

```
User types question + selects topic
        │
        ▼
POST /api/chat  { question, topic_key, top_k }
        │
        ├── RETRIEVAL PHASE
        │   ├── Load source text from data/processed/{key}_ori.txt
        │   ├── Chunk into ~150-word overlapping segments
        │   ├── Embed all chunks with MiniLM (cached after first call)
        │   ├── Embed the user's question with MiniLM
        │   └── Cosine similarity → select top-k most relevant chunks
        │
        ├── GENERATION PHASE
        │   ├── Build prompt: system instruction + retrieved chunks + question
        │   └── Send to Gemini → get grounded answer
        │
        └── Return { answer, contexts[ {text, score} ], context_relevance }
```

**Why RAG instead of sending the full lecture to Gemini:**
- Focuses the model on relevant content (better answers)
- Reduces token usage (cheaper, stays within free tier)
- Makes the system transparent (user can see which chunks were used)

### 3. RAG Quality Scoring

```
User requests RAG evaluation on a Q&A pair
        │
        ▼
POST /api/rag-score  { question, answer, contexts }
        │
        ├── Build evaluation prompt with all three metrics
        │
        ├── Single Gemini call → returns JSON scores
        │
        └── Return {
              faithfulness,        # Is the answer supported by contexts?
              context_precision,   # Are retrieved chunks relevant to the question?
              context_recall,      # Do contexts cover enough to answer fully?
              overall_score        # Weighted avg (faith=0.4, prec=0.3, rec=0.3)
            }
```

**Scoring method:** Gemini acts as an LLM judge, rating each metric 0.0–1.0.
Faithfulness is weighted highest (0.4) because avoiding hallucination matters
most in an educational tool.

### 4. Custom Text Summarization

```
User pastes custom lecture text
        │
        ▼
POST /api/summarize  { text, style }
        │
        ├── style = "concise"  → 100-300 word summary
        ├── style = "detailed" → 300-600 word structured summary
        │
        ├── Send to Gemini with appropriate prompt
        │
        └── Return { summary, word_count }
```

## Caching Strategy

| What | Where | Lifetime | Invalidation |
|------|-------|----------|-------------|
| Gemini client | `_gemini_client` | Server lifetime | Restart server |
| MiniLM model | `_embed_model` | Server lifetime | Restart server |
| Topic chunks + embeddings | `_topic_cache[key]` | Server lifetime | Restart server |
| Generated summaries | `_summary_cache[key]` | Server lifetime | `?regenerate=true` or restart |

Everything is in-memory. A server restart clears all caches, which is fine
because Gemini calls are fast (~1-2s) and the free tier has ample quota.

## How to Run

```bash
# Install dependencies
pip install -r requirements-app.txt

# Set API key
cp .env.example .env
# Edit .env → add your Gemini key from https://aistudio.google.com/apikey

# Start the server
uvicorn app.main:app --reload

# Open API docs
# http://localhost:8000/docs
```
