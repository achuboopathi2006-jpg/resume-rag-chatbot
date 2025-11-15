# bot.py
"""
Memory-safe Resume RAG with OpenAI v1 SDK (no openai.error), rate-limit handling, batching, caching.
"""

import os
import re
import uuid
import time
from typing import List, Dict

import streamlit as st
import PyPDF2
import pdfplumber
import numpy as np
from dotenv import load_dotenv
import openai

# ---- Config ----
load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_DEFAULT = 4
BATCH_SIZE = 8  # Safe batch size for memory/quota

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY not set. Add it to .env or environment.")
    st.stop()
openai.api_key = OPENAI_KEY

st.set_page_config(layout="wide", page_title="Resume RAG")
st.title("Resume RAG — OpenAI SDK v1 Compatible")

VECTOR_STORE: List[Dict] = []

# ---- Helper functions ----
def extract_text_pages_from_pypdf(pdf_bytes: bytes) -> List[str]:
    reader = PyPDF2.PdfReader(pdf_bytes)
    return [p.extract_text() or "" for p in reader.pages]

def extract_text_pages_from_pdfplumber(file_obj) -> List[str]:
    pages = []
    with pdfplumber.open(file_obj) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages

def chunk_text_generator(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        yield text[start:end].strip()
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break

def guess_name_from_text(text: str) -> str:
    m = re.search(r"(?:Name|Candidate|Candidate Name)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).splitlines()[0].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "unknown"
    first = lines[0]
    if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+", first):
        return first
    for i, ln in enumerate(lines[:8]):
        if re.search(r"Email|E-mail|Phone|Contact|Mobile", ln, flags=re.IGNORECASE):
            if i >= 1:
                return lines[i-1]
    return first[:40]

# ---- Embeddings ----
def compute_embeddings(texts: list) -> list:
    """Compute embeddings with retry for rate-limits (OpenAI SDK v1 compatible)."""
    if not texts:
        return []
    for attempt in range(5):
        try:
            resp = openai.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            # If rate limit, sleep and retry
            if hasattr(e, "http_status") and e.http_status == 429:
                st.warning(f"Rate limit hit, retrying 5s (attempt {attempt+1}/5)...")
                time.sleep(5)
            else:
                st.error(f"Embedding error: {e}")
                return []
    st.error("Exceeded rate limit. Try later or reduce PDF size.")
    return []

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_top_k(query: str, k: int = TOP_K_DEFAULT):
    q_emb = compute_embeddings([query])[0]
    q_arr = np.array(q_emb, dtype=np.float32)
    sims = [cosine_sim(q_arr, item["emb"]) for item in VECTOR_STORE]
    idx_sorted = np.argsort(sims)[::-1][:k]
    return [{"score": sims[idx], "id": VECTOR_STORE[idx]["id"], "text": VECTOR_STORE[idx]["text"], "meta": VECTOR_STORE[idx]["meta"]} for idx in idx_sorted]

def build_prompt(query: str, contexts: List[Dict]) -> str:
    intro = (
        "You are an assistant that finds candidates from resume text. "
        "Use the provided resume excerpts and answer concisely in English. "
        "If the info isn't present, say you do not find it."
    )
    ctx = ""
    for i, c in enumerate(contexts, start=1):
        name = c["meta"].get("candidate_name", "unknown")
        src = c["meta"].get("source", "")
        snippet = c["text"][:1200]
        ctx += f"\n---\n[{i}] Candidate: {name} (source: {src})\n{snippet}\n"
    return f"{intro}\n\nUser question: {query}\n\nRelevant resumes:{ctx}\n\nAnswer in English, be concise and cite which candidate you used."

# ---- Chat Completion ----
def ask_chat(prompt: str) -> str:
    try:
        resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Handle rate-limit by checking http_status == 429
        if hasattr(e, "http_status") and e.http_status == 429:
            st.error("Rate limit hit. Try again in a few seconds.")
        else:
            st.error(f"Chat error: {e}")
        return ""

# ---- Streamlit UI ----
st.sidebar.header("Controls")
top_k = st.sidebar.number_input("Top K results", min_value=1, max_value=10, value=TOP_K_DEFAULT)

uploaded = st.file_uploader("Upload bulk resume PDF", type=["pdf"])
if uploaded is not None:
    st.info("Reading PDF and extracting text...")
    try:
        pages = extract_text_pages_from_pypdf(uploaded.read())
    except Exception:
        uploaded.seek(0)
        pages = extract_text_pages_from_pdfplumber(uploaded)
    st.success(f"Extracted {len(pages)} pages.")

    joined = "\n===PAGE===\n".join(pages)
    segments = re.split(r"\n\s*(?:Resume|RESUME|Curriculum Vitae|CV)\b", joined, flags=re.IGNORECASE)
    docs = []

    if len(segments) <= 1:
        for i, p in enumerate(pages):
            docs.append({"text": p, "source": f"page_{i+1}", "candidate_name": guess_name_from_text(p)})
    else:
        for i, s in enumerate(segments):
            t = s.strip()
            if not t:
                continue
            docs.append({"text": t, "source": f"segment_{i+1}", "candidate_name": guess_name_from_text(t)})

    # ---- Chunking & Embedding ----
    st.info("Creating chunks and computing embeddings incrementally...")
    total_chunks = 0

    for doc in docs:
        chunk_gen = chunk_text_generator(doc["text"])
        batch_chunks = []
        for c in chunk_gen:
            batch_chunks.append({"id": str(uuid.uuid4()), "text": c, "meta": {"source": doc["source"], "candidate_name": doc["candidate_name"]}})
            if len(batch_chunks) >= BATCH_SIZE:
                texts = [b["text"] for b in batch_chunks]
                embs = compute_embeddings(texts)
                for b, emb in zip(batch_chunks, embs):
                    VECTOR_STORE.append({"id": b["id"], "text": b["text"], "meta": b["meta"], "emb": np.array(emb, dtype=np.float32)})
                total_chunks += len(batch_chunks)
                batch_chunks = []

        if batch_chunks:
            texts = [b["text"] for b in batch_chunks]
            embs = compute_embeddings(texts)
            for b, emb in zip(batch_chunks, embs):
                VECTOR_STORE.append({"id": b["id"], "text": b["text"], "meta": b["meta"], "emb": np.array(emb, dtype=np.float32)})
            total_chunks += len(batch_chunks)

    st.success(f"Indexing finished! Total chunks: {total_chunks}.")
    np.save("vector_store.npy", VECTOR_STORE, allow_pickle=True)

# ---- Search / Ask ----
st.markdown("---")
st.header("Search / Ask")
query = st.text_input("Enter a candidate name or ask a question", value="")

if st.button("Search") and query.strip():
    if not VECTOR_STORE:
        st.warning("No documents indexed yet. Upload a PDF first.")
    else:
        with st.spinner("Retrieving top matches..."):
            hits = retrieve_top_k(query, k=int(top_k))
        if not hits:
            st.info("No matches found.")
        else:
            st.subheader("Top retrieved snippets")
            for i, h in enumerate(hits, start=1):
                name = h["meta"].get("candidate_name", "unknown")
                src = h["meta"].get("source", "")
                st.write(f"**{i}. {name}** — source: {src} — score: {h['score']:.4f}")
                st.caption(h["text"][:800].replace("\n", " ") + ("..." if len(h["text"]) > 800 else ""))

            prompt = build_prompt(query, hits)
            answer = ask_chat(prompt)
            if answer:
                st.markdown("### Answer")
                st.write(answer)

# ---- Show indexed candidates ----
st.markdown("---")
if VECTOR_STORE:
    st.subheader("Indexed candidate examples (guesses)")
    seen = set()
    names = []
    for it in VECTOR_STORE:
        nm = it["meta"].get("candidate_name","unknown")
        if nm not in seen:
            names.append(nm)
            seen.add(nm)
        if len(names) >= 20:
            break
    for i, nm in enumerate(names, start=1):
        st.write(f"{i}. {nm}")
else:
    st.info("No resumes indexed yet. Upload a PDF to begin.")
