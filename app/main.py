"""
app/main.py — FastAPI backend for StudyLens interactive chatbot.

Endpoints:
  GET  /api/topics           List available lecture topics with pre-generated summaries.
  POST /api/summarize        Summarize user-provided lecture text via Gemini.
  POST /api/chat             RAG-based Q&A: retrieve relevant chunks → Gemini answer.
  POST /api/rag-score        Evaluate a Q&A pair on faithfulness, context precision/recall.

All endpoints return JSON.  Run with:
  uvicorn app.main:app --reload
"""

import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Resolve project root (parent of app/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoids slow startup if only some endpoints are used)
# ---------------------------------------------------------------------------
_gemini_client = None
_embed_model = None
_topic_cache = None   # {topic_key: {"chunks": [...], "embeddings": np.ndarray}}

GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Topic metadata — same as eval.py so display names stay consistent.
TOPIC_NAMES = {
    "dl_s1": "Introduction to Deep Learning",
    "dl_s2": "Computer Vision 1",
    "dl_s3": "Computer Vision 2",
    "dl_s4": "NLP 1",
    "dl_s5": "NLP 2",
    "ml_s1": "Machine Learning Intro",
    "ml_s2": "Supervised Learning",
    "ml_s3": "Unsupervised Learning",
    "ml_s4": "Evaluation & Model Selection",
    "ml_s5": "Advanced Topics in ML",
}

# Paths to source lecture data
SOURCE_DIR = PROJECT_ROOT / "data" / "processed"

# In-memory cache for Gemini-generated summaries (avoids re-generating on every request)
_summary_cache: dict = {}


def _get_gemini_client():
    """Lazily initialise the Gemini client (one instance for the whole app)."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "paste-your-key-here":
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY not configured. Get one free at https://aistudio.google.com/apikey",
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_embed_model():
    """Lazily load the sentence-transformer embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


# ---------------------------------------------------------------------------
# Chunking — reuse logic from scripts/build_features.py without importing it
# directly (avoids pulling in heavy deps like torch/transformers at import time).
# We use a lightweight word-based chunker for the web app since we don't need
# BART-tokenizer precision for retrieval.
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_words: int = 150, overlap_words: int = 20) -> List[str]:
    """Split text into overlapping word-based chunks for embedding."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start = end - overlap_words
    return chunks


# ---------------------------------------------------------------------------
# Topic loading and embedding (cached per topic)
# ---------------------------------------------------------------------------

def _load_topic_index(topic_key: str) -> dict:
    """Load source text for a topic, chunk it, and embed the chunks.

    Returns {"chunks": [str, ...], "embeddings": np.ndarray}.
    Results are cached in _topic_cache so subsequent calls are instant.
    """
    global _topic_cache
    if _topic_cache is None:
        _topic_cache = {}
    if topic_key in _topic_cache:
        return _topic_cache[topic_key]

    source_path = SOURCE_DIR / f"{topic_key}_ori.txt"
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file not found for topic {topic_key}")

    text = source_path.read_text(encoding="utf-8")
    chunks = _chunk_text(text)

    model = _get_embed_model()
    embeddings = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)

    _topic_cache[topic_key] = {"chunks": chunks, "embeddings": embeddings}
    return _topic_cache[topic_key]


def _retrieve_chunks(query: str, topic_key: str, top_k: int = 5) -> List[dict]:
    """Retrieve the top-k most relevant chunks for a query from a topic's index."""
    from sklearn.metrics.pairwise import cosine_similarity

    index = _load_topic_index(topic_key)
    model = _get_embed_model()
    q_vec = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q_vec, index["embeddings"])[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [
        {"text": index["chunks"][i], "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="StudyLens API",
    description="AI-powered lecture summarization and Q&A chatbot",
    version="1.0.0",
)

# Allow any frontend origin during development.  For production, restrict this
# to your actual deployment domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str
    style: str = "concise"  # "concise" (100-300 words) or "detailed" (300-600 words)

class SummarizeResponse(BaseModel):
    summary: str
    word_count: int

class ChatRequest(BaseModel):
    question: str
    topic_key: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    contexts: List[dict]       # retrieved chunks with similarity scores
    context_relevance: float   # average cosine similarity of retrieved chunks

class RAGScoreRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]   # the context texts that were used to generate the answer

class RAGScoreResponse(BaseModel):
    faithfulness: float        # 0-1: is the answer supported by the contexts?
    context_precision: float   # 0-1: are the contexts relevant to the question?
    context_recall: float      # 0-1: do the contexts cover what's needed to answer?
    overall_score: float       # weighted average


# ── GET /api/topics ───────────────────────────────────────────────────────────

@app.get("/api/topics")
def list_topics():
    """Return all available lecture topics.

    The frontend uses this to populate the left sidebar topic list.
    Each topic is available if its source _ori.txt file exists in data/processed/.
    """
    topics = []
    for key, name in TOPIC_NAMES.items():
        source_path = SOURCE_DIR / f"{key}_ori.txt"
        topics.append({
            "key": key,
            "name": name,
            "available": source_path.exists(),
            "has_cached_summary": key in _summary_cache,
        })
    return {"topics": topics}


# ── POST /api/summarize ──────────────────────────────────────────────────────

@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize_text(req: SummarizeRequest):
    """Summarize user-provided lecture text using Gemini.

    Why Gemini Flash: Our experiments showed LLMs (Qwen) outperformed all smaller
    models on both ROUGE-L and BERTScore.  Gemini Flash is a comparable LLM
    available on a free tier — no GPU hosting required.

    The prompt mirrors the one used in GeminiSummarizer (scripts/model.py) so
    the web app produces summaries consistent with our evaluated pipeline.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    client = _get_gemini_client()

    if req.style == "detailed":
        prompt = (
            "You are an expert educational summarizer. "
            "Create a detailed, structured summary of the following lecture content. "
            "Cover all major topics and subtopics with key details. "
            "Use clear section headers. Be factual — only include information "
            "present in the source material. "
            "Aim for 300-600 words.\n\n"
            f"LECTURE CONTENT:\n{req.text}"
        )
    else:
        prompt = (
            "You are an expert educational summarizer. "
            "Summarize the following lecture content into a concise, coherent summary "
            "suitable for a student reviewing for exams. "
            "Focus on key concepts, definitions, methods, and relationships. "
            "Be factual — only include information present in the source material. "
            "Keep the summary between 100-300 words.\n\n"
            f"LECTURE CONTENT:\n{req.text}"
        )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    summary = response.text
    return SummarizeResponse(summary=summary, word_count=len(summary.split()))


# ── POST /api/chat ───────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """RAG-based chatbot: retrieve relevant lecture chunks, then answer.

    How it works (Retrieval-Augmented Generation):
    1. User asks a question and selects a topic.
    2. We load the topic's source text (from data/processed/*_ori.txt),
       chunk it, and embed each chunk with MiniLM (sentence-transformers).
    3. We embed the user's question with the same model.
    4. Cosine similarity finds the top-k most relevant chunks.
    5. Those chunks are injected into the Gemini prompt as context.
    6. Gemini generates an answer grounded in the retrieved context.

    Why RAG instead of stuffing the entire lecture into the prompt:
    - Focuses the model on the most relevant content
    - Reduces token usage (cheaper, faster)
    - Makes the system transparent — users can see which chunks were used
    """
    if req.topic_key not in TOPIC_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown topic: {req.topic_key}")

    # Step 1-4: Retrieve relevant chunks
    retrieved = _retrieve_chunks(req.question, req.topic_key, top_k=req.top_k)

    # Step 5: Build context-augmented prompt
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}, relevance={c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    prompt = (
        "You are a helpful teaching assistant for a university course. "
        "Answer the student's question using ONLY the lecture context provided below. "
        "If the context does not contain enough information, say so honestly. "
        "Be clear, accurate, and cite specific details from the context.\n\n"
        f"LECTURE CONTEXT:\n{context_block}\n\n"
        f"STUDENT QUESTION: {req.question}"
    )

    # Step 6: Generate answer
    client = _get_gemini_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    answer = response.text

    # Context relevance = average cosine similarity of retrieved chunks
    avg_relevance = round(
        sum(c["score"] for c in retrieved) / len(retrieved) if retrieved else 0.0,
        4,
    )

    return ChatResponse(
        answer=answer,
        contexts=retrieved,
        context_relevance=avg_relevance,
    )


# ── POST /api/rag-score ──────────────────────────────────────────────────────

@app.post("/api/rag-score", response_model=RAGScoreResponse)
def compute_rag_score(req: RAGScoreRequest):
    """Evaluate RAG quality: faithfulness, context precision, context recall.

    Why a separate endpoint: Computing these scores requires an extra LLM call
    (Gemini judges its own output).  Making it opt-in keeps the /chat endpoint
    fast while still giving users transparency into RAG quality when they want it.

    How each metric is computed:
    - Faithfulness:        "Is the answer supported by the provided contexts?"
                           Gemini rates 0-1.  High = answer doesn't hallucinate.
    - Context Precision:   "Are the retrieved contexts relevant to the question?"
                           Gemini rates 0-1.  High = retrieval found good chunks.
    - Context Recall:      "Do the contexts contain enough info to fully answer?"
                           Gemini rates 0-1.  High = no important info was missed.
    - Overall Score:       Weighted average (faithfulness=0.4, precision=0.3, recall=0.3)
                           because faithfulness (no hallucination) matters most.
    """
    if not req.contexts:
        raise HTTPException(status_code=400, detail="Contexts list cannot be empty")

    client = _get_gemini_client()
    context_block = "\n\n---\n\n".join(req.contexts)

    # Single Gemini call to evaluate all three metrics at once (cheaper than 3 calls)
    eval_prompt = (
        "You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. "
        "Evaluate the following Q&A pair and its retrieved contexts.\n\n"
        f"QUESTION: {req.question}\n\n"
        f"ANSWER: {req.answer}\n\n"
        f"RETRIEVED CONTEXTS:\n{context_block}\n\n"
        "Rate each metric from 0.0 to 1.0:\n"
        "1. faithfulness: Is the answer factually supported by the contexts? "
        "(1.0 = fully supported, 0.0 = completely hallucinated)\n"
        "2. context_precision: Are the retrieved contexts relevant to the question? "
        "(1.0 = all highly relevant, 0.0 = none relevant)\n"
        "3. context_recall: Do the contexts contain enough information to fully answer the question? "
        "(1.0 = complete coverage, 0.0 = no useful information)\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"faithfulness": 0.X, "context_precision": 0.X, "context_recall": 0.X}'
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=eval_prompt)
    raw = response.text.strip()

    # Parse the JSON response — handle markdown code fences if Gemini wraps it
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned invalid JSON for RAG evaluation: {raw[:200]}",
        )

    faith = max(0.0, min(1.0, float(scores.get("faithfulness", 0.0))))
    prec = max(0.0, min(1.0, float(scores.get("context_precision", 0.0))))
    recall = max(0.0, min(1.0, float(scores.get("context_recall", 0.0))))
    overall = round(0.4 * faith + 0.3 * prec + 0.3 * recall, 4)

    return RAGScoreResponse(
        faithfulness=round(faith, 4),
        context_precision=round(prec, 4),
        context_recall=round(recall, 4),
        overall_score=overall,
    )


# ── GET /api/topic/{topic_key}/summary ────────────────────────────────────────

@app.get("/api/topic/{topic_key}/summary")
def get_topic_summary(topic_key: str, regenerate: bool = False):
    """Generate a summary for a topic's lecture material using Gemini.

    Reads the source _ori.txt (slides + transcript + notes combined), sends it
    to Gemini Flash, and returns a structured summary.  Results are cached in
    memory so subsequent requests for the same topic are instant.

    Pass ?regenerate=true to force a fresh Gemini call (e.g. if the user wants
    a different summary).
    """
    if topic_key not in TOPIC_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown topic: {topic_key}")

    # Return cached summary if available (unless regenerate requested)
    if not regenerate and topic_key in _summary_cache:
        return _summary_cache[topic_key]

    source_path = SOURCE_DIR / f"{topic_key}_ori.txt"
    if not source_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source file not found for {topic_key}. Run the data pipeline first.",
        )

    source_text = source_path.read_text(encoding="utf-8")

    client = _get_gemini_client()
    prompt = (
        "You are an expert educational summarizer for university courses. "
        "Summarize the following lecture material into a well-structured summary "
        "suitable for a student reviewing for exams.\n\n"
        "Your summary should:\n"
        "- Start with a one-line title of the topic\n"
        "- List key concepts covered\n"
        "- Include important definitions, formulas, and technical details\n"
        "- Preserve the logical flow of the lecture\n"
        "- Use clear section headers and bullet points\n"
        "- Be 300-500 words\n\n"
        "Be factual — only include information present in the source material.\n\n"
        f"LECTURE MATERIAL:\n{source_text}"
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    summary = response.text

    result = {
        "topic_key": topic_key,
        "topic_name": TOPIC_NAMES[topic_key],
        "model": "Gemini 2.0 Flash",
        "summary": summary,
        "word_count": len(summary.split()),
    }

    # Cache so we don't re-generate on every page load
    _summary_cache[topic_key] = result
    return result


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check for uptime monitors (e.g., UptimeRobot)."""
    return {"status": "ok", "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))}
