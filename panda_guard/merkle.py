"""
panda_guard/merkle.py

Binary Merkle Tree for embedding vector integrity verification.

"Pet Panda" — ensures all higher-dimensional embedding vectors are
air-gapped or hashed into binary for the Merkle tree so no sensitive
data is uploaded and no needed context is removed.

The Merkle tree is built over SHA-256 hashes of binary-serialized
float32 vectors. Only hashes are stored and transmitted — original
vectors never leave the local machine.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Merkle Tree ───────────────────────────────────────────────────────────────

@dataclass
class MerkleProof:
    """Inclusion proof for a leaf in the Merkle tree."""
    leaf_index: int
    leaf_hash: str
    proof_hashes: list[str]
    proof_directions: list[str]  # "left" or "right"
    root_hash: str


@dataclass
class MerkleNode:
    """A node in the binary Merkle tree."""
    hash_value: str
    left: MerkleNode | None = None
    right: MerkleNode | None = None
    leaf_index: int = -1


class MerkleTree:
    """
    Binary Merkle tree built over embedding vectors.

    Each leaf is SHA-256(binary_repr(float32_vector)), ensuring:
    1. Original float vectors are never stored in the tree
    2. Tree can verify membership without revealing vector content
    3. Any tampered vector produces a different root hash

    Usage:
        vectors = np.random.randn(100, 1536).astype(np.float32)
        tree = MerkleTree.from_vectors(vectors)
        proof = tree.get_proof(42)
        assert MerkleTree.verify_proof(proof)
    """

    def __init__(self, leaf_hashes: list[str]) -> None:
        self._leaf_hashes = leaf_hashes
        self._root = self._build_tree(leaf_hashes)

    @classmethod
    def from_vectors(cls, vectors: np.ndarray) -> MerkleTree:
        """
        Build a Merkle tree from a matrix of embedding vectors.

        Args:
            vectors: (N, D) float32 matrix of embedding vectors

        Returns:
            MerkleTree with SHA-256 hashed binary leaves
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        leaf_hashes = [hash_vector(v) for v in vectors]
        return cls(leaf_hashes)

    @classmethod
    def from_npz(cls, npz_path: str) -> MerkleTree:
        """Build a Merkle tree from a saved .npz embedding store."""
        data = np.load(npz_path, allow_pickle=True)
        vectors = data["vectors"]
        return cls.from_vectors(vectors)

    @property
    def root_hash(self) -> str:
        """The root hash — integrity seal for the entire embedding store."""
        return self._root.hash_value if self._root else ""

    @property
    def leaf_count(self) -> int:
        """Number of leaves (vectors) in the tree."""
        return len(self._leaf_hashes)

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """
        Generate an inclusion proof for a leaf.

        The proof allows verifying that a specific vector is part of
        the embedding store without revealing any other vectors.
        """
        if leaf_index < 0 or leaf_index >= len(self._leaf_hashes):
            raise IndexError(
                f"Leaf index {leaf_index} out of range [0, {len(self._leaf_hashes)})"
            )

        proof_hashes: list[str] = []
        proof_directions: list[str] = []

        # Walk up the tree collecting sibling hashes
        nodes = self._leaf_hashes[:]
        # Pad to power of 2
        while len(nodes) & (len(nodes) - 1):
            nodes.append(nodes[-1])

        idx = leaf_index
        while len(nodes) > 1:
            next_level: list[str] = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left

                if i == idx or i + 1 == idx:
                    if i == idx:
                        proof_hashes.append(right)
                        proof_directions.append("right")
                    else:
                        proof_hashes.append(left)
                        proof_directions.append("left")

                combined = _hash_pair(left, right)
                next_level.append(combined)

            idx = idx // 2
            nodes = next_level

        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=self._leaf_hashes[leaf_index],
            proof_hashes=proof_hashes,
            proof_directions=proof_directions,
            root_hash=self.root_hash,
        )

    @staticmethod
    def verify_proof(proof: MerkleProof) -> bool:
        """
        Verify a Merkle inclusion proof.

        Returns True if the proof is valid — the leaf hash, combined
        with the sibling hashes in the proof path, produces the root hash.
        """
        current = proof.leaf_hash

        for sibling_hash, direction in zip(
            proof.proof_hashes, proof.proof_directions
        ):
            if direction == "left":
                current = _hash_pair(sibling_hash, current)
            else:
                current = _hash_pair(current, sibling_hash)

        return current == proof.root_hash

    def verify_vector(self, vector: np.ndarray, leaf_index: int) -> bool:
        """
        Verify that a specific vector matches its leaf in the tree.

        This is the air-gap check: compute the hash of the vector
        and compare it to the stored leaf hash.
        """
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        computed_hash = hash_vector(vector)
        return computed_hash == self._leaf_hashes[leaf_index]

    def _build_tree(self, leaf_hashes: list[str]) -> MerkleNode | None:
        """Build the binary tree from leaf hashes."""
        if not leaf_hashes:
            return None

        # Create leaf nodes
        nodes = [
            MerkleNode(hash_value=h, leaf_index=i)
            for i, h in enumerate(leaf_hashes)
        ]

        # Pad to power of 2
        while len(nodes) & (len(nodes) - 1):
            nodes.append(MerkleNode(hash_value=nodes[-1].hash_value))

        # Build tree bottom-up
        while len(nodes) > 1:
            next_level: list[MerkleNode] = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left
                parent = MerkleNode(
                    hash_value=_hash_pair(left.hash_value, right.hash_value),
                    left=left,
                    right=right,
                )
                next_level.append(parent)
            nodes = next_level

        return nodes[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tree metadata (not the vectors!) for storage."""
        return {
            "root_hash": self.root_hash,
            "leaf_count": self.leaf_count,
            "leaf_hashes": self._leaf_hashes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MerkleTree:
        """Reconstruct from serialized metadata."""
        return cls(data["leaf_hashes"])


# ── Hash Utilities ────────────────────────────────────────────────────────────

def hash_vector(vector: np.ndarray) -> str:
    """
    Hash a float32 vector into a SHA-256 hex digest.

    The vector is serialized to its raw binary representation
    (IEEE 754 float32 bytes), then hashed. This ensures:
    - Deterministic hashing (same vector → same hash)
    - No float precision issues (binary, not text)
    - No sensitive data in the hash (one-way function)
    """
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    binary = vector.tobytes()
    return hashlib.sha256(binary).hexdigest()


def hash_text(text: str) -> str:
    """Hash text content using SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    """Combine two hex hashes into a parent hash."""
    combined = (left + right).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()
