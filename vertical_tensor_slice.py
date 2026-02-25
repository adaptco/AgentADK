"""
rag/vertical_tensor_slice.py

Ingests the A2A_MCP monorepo into a d=1536 L2-normalised embedding matrix.

"Vertical tensor slice" means: for a given agent query, we extract one
column of the repo-tensor — the projection of every file chunk onto the
query direction. Only the top-k projections are returned. This is:
  1. Normalized dot product (NDP) — faster than cosine
  2. Semantically equivalent to cosine similarity when vectors are L2-normalised
  3. "Vertical" because we slice across ALL files on a SINGLE query axis

Usage:
    python rag/vertical_tensor_slice.py --repo /path/to/A2A_MCP --out rag/embedding_store.npz
    python rag/vertical_tensor_slice.py --query "how does IntentEngine route tasks"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
EMBED_DIM = 1536
CHUNK_SIZE = 400          # tokens (approx characters / 4)
CHUNK_OVERLAP = 80        # overlap in characters for context continuity
INCLUDE_EXTENSIONS = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt"}
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", "node_modules", "dist", ".mypy_cache",
    "*.egg-info", ".pytest_cache", "frontend",
}

REPO_ROOT = Path(os.environ.get("A2A_REPO_ROOT", Path(__file__).parent.parent))


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_file(path: Path, chunk_chars: int = CHUNK_SIZE * 4) -> Iterator[tuple[str, str]]:
    """
    Yield (chunk_key, chunk_text) pairs for a file.
    chunk_key = "<relative_path>:<start_char>"
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    if len(text.strip()) < 20:
        return

    rel = str(path.relative_to(REPO_ROOT))
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end]
        # Include file path as context header for better retrieval
        key = f"{rel}:{start}"
        yield key, f"# FILE: {rel}\n{chunk}"
        start += chunk_chars - (CHUNK_OVERLAP * 4)


def iter_repo_files(repo: Path) -> Iterator[Path]:
    """Walk the repo, skipping excluded dirs, yielding eligible source files."""
    for path in sorted(repo.rglob("*")):
        if any(ex in path.parts for ex in EXCLUDE_DIRS):
            continue
        if path.suffix in INCLUDE_EXTENSIONS and path.is_file():
            yield path


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts using OpenAI text-embedding-3-small.
    Falls back to deterministic hash embeddings for offline use.
    Returns L2-normalised (N, 1536) float32 matrix.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vecs = np.array([r.embedding for r in resp.data], dtype=np.float32)
    except Exception as e:
        print(f"  [offline fallback] {e}")
        vecs = _hash_embed_batch(texts)

    # L2-normalise each row — after this, dot(a, b) == cosine(a, b)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _hash_embed_batch(texts: list[str]) -> np.ndarray:
    """Deterministic offline fallback — NOT semantically meaningful."""
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        raw = hashlib.sha256(text.encode()).digest() * (EMBED_DIM // 32 + 1)
        vec = np.frombuffer(raw[: EMBED_DIM * 4], dtype=np.float32).copy()
        out[i] = vec
    return out


# ── Build store ───────────────────────────────────────────────────────────────
def build_embedding_store(
    repo: Path = REPO_ROOT,
    out_path: Path = REPO_ROOT / "rag" / "embedding_store.npz",
    batch_size: int = 64,
) -> None:
    """
    Ingest the full repo → chunk → embed → L2-normalise → save as .npz

    The .npz file contains three parallel arrays:
        keys    (N,)        str   — chunk key "<file>:<offset>"
        vectors (N, 1536)   f32   — L2-normalised embedding matrix
        texts   (N,)        str   — raw chunk text (for display)
    """
    print(f"Ingesting repo: {repo}")
    all_keys: list[str] = []
    all_texts: list[str] = []

    for file_path in iter_repo_files(repo):
        for key, text in chunk_file(file_path):
            all_keys.append(key)
            all_texts.append(text)

    print(f"  Total chunks: {len(all_keys)}")

    all_vectors: list[np.ndarray] = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i: i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(all_texts) - 1) // batch_size + 1}...")
        vecs = embed_batch(batch)
        all_vectors.append(vecs)

    matrix = np.vstack(all_vectors)  # (N, 1536)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keys=np.array(all_keys),
        vectors=matrix,
        texts=np.array(all_texts),
    )
    print(f"  Saved: {out_path}  shape={matrix.shape}")


# ── Vertical Tensor Slice Query ───────────────────────────────────────────────
class VerticalTensorSlicer:
    """
    Given a pre-built embedding store, performs vertical tensor slicing:

    For query q (d=1536, L2-normalised):
        scores = M @ q           # M is (N, 1536) corpus matrix
        slice  = top_k(scores)   # vertical column slice

    This is the normalized dot product. When M rows and q are L2-normalised,
    this equals cosine similarity but is faster (no per-pair norm computation).
    """

    def __init__(self, store_path: Path = REPO_ROOT / "rag" / "embedding_store.npz"):
        data = np.load(store_path, allow_pickle=True)
        self.keys: list[str] = data["keys"].tolist()
        self.matrix: np.ndarray = data["vectors"]    # (N, 1536), already L2-normalised
        self.texts: list[str] = data["texts"].tolist()
        print(f"[VTS] Loaded {self.matrix.shape[0]} chunks from {store_path}")

    def query(self, query_text: str, top_k: int = 8, agent_filter: str | None = None) -> list[dict]:
        """
        Slice the tensor on the query axis.

        agent_filter: if set, only return chunks from files under this path prefix
                      (e.g. "orchestrator/" returns only orchestrator context)
        """
        q = embed_batch([query_text])[0]  # (1536,), L2-normalised

        # Vertical slice — one matrix multiply
        scores = self.matrix @ q  # (N,)

        if agent_filter:
            # Zero out scores for chunks not in this agent's domain
            mask = np.array([agent_filter in k for k in self.keys])
            scores = np.where(mask, scores, -np.inf)

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {
                "key": self.keys[i],
                "file": self.keys[i].split(":")[0],
                "score": float(scores[i]),
                "text": self.texts[i][:500],
            }
            for i in top_idx
            if scores[i] > -np.inf
        ]

    def agent_capability_vector(self, agent_system_prompt: str) -> np.ndarray:
        """
        Compute the capability vector for an agent.
        Used for semantic routing: route message to agent whose capability
        vector has highest NDP with the message embedding.
        """
        return embed_batch([agent_system_prompt])[0]

    def route_to_agent(
        self,
        message: str,
        agent_prompts: dict[str, str],
    ) -> tuple[str, float]:
        """
        Given a message and a dict of {agent_id: system_prompt},
        return the (agent_id, score) of the best-matching agent.
        Uses NDP between message embedding and agent capability vectors.
        """
        msg_vec = embed_batch([message])[0]
        best_agent, best_score = "", -1.0

        for agent_id, prompt in agent_prompts.items():
            agent_vec = self.agent_capability_vector(prompt)
            score = float(np.dot(msg_vec, agent_vec))
            if score > best_score:
                best_score, best_agent = score, agent_id

        return best_agent, best_score


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Vertical Tensor Slice — RAG for A2A_MCP")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--out", default="rag/embedding_store.npz")
    parser.add_argument("--query", default="", help="Run a query against the built store")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--agent-filter", default="", help="Restrict results to a subdirectory")
    parser.add_argument("--build", action="store_true", help="Rebuild the embedding store")
    args = parser.parse_args()

    if args.build or not Path(args.out).exists():
        build_embedding_store(repo=Path(args.repo), out_path=Path(args.out))

    if args.query:
        slicer = VerticalTensorSlicer(Path(args.out))
        results = slicer.query(
            args.query,
            top_k=args.top_k,
            agent_filter=args.agent_filter or None,
        )
        print(f"\nQuery: {args.query}\n{'─' * 60}")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['file']}")
            print(f"          {r['text'][:120].strip()}")
            print()


if __name__ == "__main__":
    main()
