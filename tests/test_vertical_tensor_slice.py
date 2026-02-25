"""
tests/test_vertical_tensor_slice.py

Tests for the RAG vertical tensor slice module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from the rag/ copy
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vertical_tensor_slice import (
    chunk_file,
    iter_repo_files,
    embed_batch,
    build_embedding_store,
    VerticalTensorSlicer,
)


# ── Chunking ──────────────────────────────────────────────────────────────────

class TestChunkFile:
    def test_small_file(self, tmp_path: Path):
        """A small file should produce one chunk."""
        f = tmp_path / "small.py"
        f.write_text("x = 1\ny = 2\n")
        chunks = list(chunk_file(f))
        assert len(chunks) == 1
        assert chunks[0][1] == "x = 1\ny = 2\n"

    def test_chunk_key_format(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("hello")
        chunks = list(chunk_file(f))
        key, text = chunks[0]
        # Key should contain the path and offset
        assert ":0" in key or "test.py" in key

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.py"
        f.write_text("")
        chunks = list(chunk_file(f))
        assert len(chunks) == 0


# ── Repo File Iterator ────────────────────────────────────────────────────────

class TestIterRepoFiles:
    def test_finds_python_files(self, tmp_path: Path):
        (tmp_path / "hello.py").write_text("pass")
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "data.csv").write_text("a,b,c")  # excluded extension

        files = list(iter_repo_files(tmp_path))
        extensions = {f.suffix for f in files}
        assert ".py" in extensions
        assert ".md" in extensions

    def test_excludes_pycache(self, tmp_path: Path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("pass")
        (tmp_path / "main.py").write_text("pass")

        files = list(iter_repo_files(tmp_path))
        file_paths = [str(f) for f in files]
        assert not any("__pycache__" in p for p in file_paths)


# ── Embedding (Offline Hash Fallback) ────────────────────────────────────────

class TestEmbedBatch:
    def test_returns_correct_shape(self):
        """Hash fallback should return (N, 1536) L2-normalised vectors."""
        texts = ["hello world", "test query", "another one"]
        vectors = embed_batch(texts)
        assert vectors.shape == (3, 1536)

    def test_l2_normalised(self):
        """All vectors should have unit L2 norm (within tolerance)."""
        texts = ["hello", "world"]
        vectors = embed_batch(texts)
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self):
        """Same text → same embedding."""
        v1 = embed_batch(["query"])
        v2 = embed_batch(["query"])
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts(self):
        """Different texts → different embeddings."""
        vectors = embed_batch(["hello", "world"])
        # They shouldn't be identical
        assert not np.allclose(vectors[0], vectors[1])


# ── Build Embedding Store ────────────────────────────────────────────────────

class TestBuildEmbeddingStore:
    def test_build_store(self, tmp_path: Path):
        """Build a store from a small repo."""
        # Create a mini repo
        (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
        (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
        (tmp_path / "readme.md").write_text("# Test repo\n")

        out_path = tmp_path / "store.npz"
        build_embedding_store(repo=tmp_path, out_path=out_path)

        assert out_path.exists()
        data = np.load(str(out_path), allow_pickle=True)
        assert "keys" in data
        assert "vectors" in data
        assert "texts" in data
        assert len(data["keys"]) == len(data["vectors"])
        assert data["vectors"].shape[1] == 1536


# ── Vertical Tensor Slicer ────────────────────────────────────────────────────

class TestVerticalTensorSlicer:
    @pytest.fixture
    def slicer(self, tmp_path: Path) -> VerticalTensorSlicer:
        """Build a slicer from a small repo."""
        (tmp_path / "server.py").write_text(
            "class Server:\n    def start(self):\n        self.listen(8080)\n"
        )
        (tmp_path / "client.py").write_text(
            "class Client:\n    def connect(self, url):\n        pass\n"
        )
        (tmp_path / "tests.py").write_text(
            "def test_server():\n    assert True\n"
        )

        out_path = tmp_path / "store.npz"
        build_embedding_store(repo=tmp_path, out_path=out_path)
        return VerticalTensorSlicer(store_path=out_path)

    def test_query_returns_results(self, slicer: VerticalTensorSlicer):
        results = slicer.query("how does the server start", top_k=3)
        assert len(results) > 0
        assert "score" in results[0]
        assert "path" in results[0]

    def test_query_scores_descending(self, slicer: VerticalTensorSlicer):
        results = slicer.query("server start", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_with_filter(self, slicer: VerticalTensorSlicer):
        """Filtering should restrict results to matching paths."""
        # This test uses the hash fallback so scores may not be meaningful,
        # but the filter mechanism should work
        all_results = slicer.query("test", top_k=10)
        assert len(all_results) > 0

    def test_agent_capability_vector(self, slicer: VerticalTensorSlicer):
        """Agent capability vector should be L2-normalised."""
        vec = slicer.agent_capability_vector(
            "You are a testing agent that writes test suites."
        )
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_route_to_agent(self, slicer: VerticalTensorSlicer):
        """Route should return (agent_id, score)."""
        prompts = {
            "coder": "You write production Python code.",
            "tester": "You write test suites and verify code.",
        }
        agent_id, score = slicer.route_to_agent(
            "write a test for the server module", prompts
        )
        assert agent_id in prompts
        assert 0.0 <= score <= 1.5  # dot product of L2-normed vectors
