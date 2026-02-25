"""
tests/test_merkle.py

Tests for the Panda Guard Merkle tree.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pytest

from panda_guard.merkle import MerkleTree, MerkleProof, hash_vector, hash_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_vectors() -> np.ndarray:
    """Deterministic sample vectors for testing."""
    rng = np.random.RandomState(42)
    return rng.randn(16, 1536).astype(np.float32)


@pytest.fixture
def tree(sample_vectors: np.ndarray) -> MerkleTree:
    return MerkleTree.from_vectors(sample_vectors)


@pytest.fixture
def small_tree() -> MerkleTree:
    """Tiny 4-vector tree for easier debugging."""
    vectors = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ], dtype=np.float32)
    return MerkleTree.from_vectors(vectors)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_from_vectors(self, tree: MerkleTree, sample_vectors: np.ndarray):
        """Tree should have correct leaf count."""
        assert tree.leaf_count == len(sample_vectors)
        assert tree.root_hash != ""

    def test_root_hash_deterministic(self, sample_vectors: np.ndarray):
        """Same vectors → same root hash."""
        tree1 = MerkleTree.from_vectors(sample_vectors)
        tree2 = MerkleTree.from_vectors(sample_vectors)
        assert tree1.root_hash == tree2.root_hash

    def test_different_vectors_different_root(self, sample_vectors: np.ndarray):
        """Different vectors → different root hash."""
        modified = sample_vectors.copy()
        modified[0, 0] += 0.001
        tree1 = MerkleTree.from_vectors(sample_vectors)
        tree2 = MerkleTree.from_vectors(modified)
        assert tree1.root_hash != tree2.root_hash

    def test_empty_tree(self):
        """Empty leaf list should produce empty root."""
        tree = MerkleTree([])
        assert tree.root_hash == ""
        assert tree.leaf_count == 0

    def test_single_leaf(self):
        """Single vector tree should work."""
        vec = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        tree = MerkleTree.from_vectors(vec)
        assert tree.leaf_count == 1
        assert tree.root_hash != ""


# ── Proofs ────────────────────────────────────────────────────────────────────

class TestProofs:
    def test_valid_proof(self, tree: MerkleTree):
        """Proof for any leaf should verify."""
        for i in range(tree.leaf_count):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof), f"Proof failed for leaf {i}"

    def test_proof_structure(self, tree: MerkleTree):
        """Proof should have correct fields."""
        proof = tree.get_proof(0)
        assert proof.leaf_index == 0
        assert proof.leaf_hash != ""
        assert proof.root_hash == tree.root_hash
        assert len(proof.proof_hashes) > 0
        assert len(proof.proof_hashes) == len(proof.proof_directions)
        assert all(d in ("left", "right") for d in proof.proof_directions)

    def test_tampered_proof_fails(self, tree: MerkleTree):
        """Modifying the leaf hash should invalidate the proof."""
        proof = tree.get_proof(0)
        tampered = MerkleProof(
            leaf_index=proof.leaf_index,
            leaf_hash="0000" + proof.leaf_hash[4:],  # tamper
            proof_hashes=proof.proof_hashes,
            proof_directions=proof.proof_directions,
            root_hash=proof.root_hash,
        )
        assert not MerkleTree.verify_proof(tampered)

    def test_out_of_range_proof(self, tree: MerkleTree):
        """Out-of-range index should raise."""
        with pytest.raises(IndexError):
            tree.get_proof(tree.leaf_count)
        with pytest.raises(IndexError):
            tree.get_proof(-1)

    def test_small_tree_proofs(self, small_tree: MerkleTree):
        """All proofs in a 4-element tree should verify."""
        for i in range(4):
            proof = small_tree.get_proof(i)
            assert MerkleTree.verify_proof(proof)

    def test_manual_construction_verify_proof(self):
        """Manually construct a tree and proof to verify logic independently."""
        # 1. Create leaf hashes
        l1 = hash_text("leaf1")
        l2 = hash_text("leaf2")
        l3 = hash_text("leaf3")
        l4 = hash_text("leaf4")

        # 2. Manually compute parent hashes (simulating _hash_pair)
        def manual_hash_pair(left, right):
            return hashlib.sha256((left + right).encode("utf-8")).hexdigest()

        # Tree structure:
        #       Root
        #      /    \
        #    H12    H34
        #    / \    / \
        #   L1 L2  L3 L4
        h12 = manual_hash_pair(l1, l2)
        h34 = manual_hash_pair(l3, l4)
        root = manual_hash_pair(h12, h34)

        # 3. Construct proof for l3 (index 2)
        # Path: l3 -> h34 -> root
        # Siblings: l4 (right of l3), h12 (left of h34)
        proof = MerkleProof(
            leaf_index=2,
            leaf_hash=l3,
            proof_hashes=[l4, h12],
            proof_directions=["right", "left"],
            root_hash=root,
        )

        # 4. Verify
        assert MerkleTree.verify_proof(proof)


# ── Tamper Detection ─────────────────────────────────────────────────────────

class TestTamperDetection:
    def test_verify_vector_match(self, sample_vectors: np.ndarray, tree: MerkleTree):
        """Correct vector should verify against its leaf."""
        assert tree.verify_vector(sample_vectors[0], 0)
        assert tree.verify_vector(sample_vectors[5], 5)

    def test_verify_vector_mismatch(self, sample_vectors: np.ndarray, tree: MerkleTree):
        """Modified vector should fail verification."""
        tampered = sample_vectors[0].copy()
        tampered[0] += 1.0
        assert not tree.verify_vector(tampered, 0)

    def test_verify_wrong_index(self, sample_vectors: np.ndarray, tree: MerkleTree):
        """Vector at wrong index should fail."""
        assert not tree.verify_vector(sample_vectors[0], 1)


# ── Serialization ─────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_and_back(self, tree: MerkleTree):
        """Round-trip through dict serialization."""
        data = tree.to_dict()
        reconstructed = MerkleTree.from_dict(data)
        assert reconstructed.root_hash == tree.root_hash
        assert reconstructed.leaf_count == tree.leaf_count

    def test_dict_has_no_vectors(self, tree: MerkleTree):
        """Serialized dict should only contain hashes, not float data."""
        data = tree.to_dict()
        serialized = str(data)
        # Should not contain any floats (only hex hashes)
        assert "array" not in serialized.lower()


# ── Hash Utilities ────────────────────────────────────────────────────────────

class TestHashUtils:
    def test_hash_vector_deterministic(self):
        """Same vector → same hash."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert hash_vector(vec) == hash_vector(vec)

    def test_hash_vector_different(self):
        """Different vectors → different hashes."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.1], dtype=np.float32)
        assert hash_vector(v1) != hash_vector(v2)

    def test_hash_vector_length(self):
        """SHA-256 hex digest should be 64 characters."""
        vec = np.array([1.0], dtype=np.float32)
        assert len(hash_vector(vec)) == 64

    def test_hash_text(self):
        """Text hashing should be deterministic."""
        assert hash_text("hello") == hash_text("hello")
        assert hash_text("hello") != hash_text("world")
        assert len(hash_text("test")) == 64
