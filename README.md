# AgentADK (A2A Digital Twin)

## What Is A2A_MCP?
A2A_MCP in this repository is a constitutional, deterministic control plane for agent-assisted software workflows, where state transitions and artifacts are validated through explicit, auditable surfaces rather than unrestricted agent autonomy.

This is a repo-grounded implementation. It includes concrete governance and lineage mechanisms that exist in this codebase today, and it documents target-state controls separately when they are not yet implemented here.

## Constitutional Invariants Enforced Here
- Deterministic state evolution: Digital Twin state is stored in `digital_twin/twin_state.json` and updated through explicit registry operations in `digital_twin/twin_registry.py`, with CI status updates persisted and replayable.
- Content-addressed lineage: Embedding integrity uses SHA-256 hashing and Merkle roots/proofs in `panda_guard/merkle.py`, including vector-level verification and inclusion proofs.
- Governed execution surfaces: MCP tools are registered through `bootstrap_digital_twin.py` and `mcp_extensions/claude_code_mcp_server.py`; CI gates are enforced through `.github/workflows/pr_validation.yml` and `a2a_twin_pipeline.yml`.
- Auditability: Workflow outputs, test reports, coverage summaries, and Twin state snapshots are written as machine-readable artifacts in CI and the repository state surfaces.

## Governance Corridor (Current Implementation)
The current corridor is governance-by-validated-surfaces:
- Requests enter through MCP tools and orchestration entrypoints.
- Agent work is constrained to registered tool calls and repository mutations.
- CI workflows validate syntax, tests, security scanning, and integrity checks.
- Twin state and CI status are written back as explicit system state.

This implementation is governed and deterministic at the integration surfaces listed above. It is not a free-form autonomous loop where agents can bypass validation gates.

## Anti-Drift and Anti-Corruption Boundaries
- MCP gateway boundary: Tool registration and invocation boundaries separate model decisions from direct repo mutation paths.
- Air-gap and scan boundary: `panda_guard/air_gap.py` scans chunk text for sensitive material and supports redaction before artifact movement.
- Integrity boundary: Merkle root/proof verification detects artifact tampering across embedding vectors.
- CI gate boundary: PR and pipeline workflows enforce repeatable checks before state advancement.

## What Exists vs What Is Target-State
| Implemented in this repo | Not implemented yet / roadmap |
| --- | --- |
| Digital Twin JSON state registry with explicit CI updates (`digital_twin/twin_registry.py`) | Formal Runtime Envelope tokening such as `0x_ENVELOPE_V4_ALPHA_STABLE` |
| SHA-256 hashing and Merkle lineage verification (`panda_guard/merkle.py`) | `LIFT_VECTOR_ALPHA` and `LIFT_VECTOR_BETA` enforcement surfaces |
| Air-gap secret scanning and optional redaction (`panda_guard/air_gap.py`) | `HAMILTONIAN_VALIDATOR` policy token integration |
| MCP tool gateway and orchestration bootstrap (`bootstrap_digital_twin.py`, `mcp_extensions/claude_code_mcp_server.py`) | `kernel.index.freeze.completed` lifecycle gates |
| CI policy gates for lint/tests/security checks (`.github/workflows/pr_validation.yml`) | Explicit `session_id` continuity contracts across all corridor phases |
| Full-pipeline orchestration with Twin and Airtable synchronization (`a2a_twin_pipeline.yml`) | RLS-backed data-plane constraints for multi-tenant governance |

## Operator Mental Model
1. A user request is received through the MCP-facing control surface.
2. Tools and agents execute within registered capabilities against repo state.
3. CI pipelines validate outcomes, integrity checks, and test status.
4. Twin state is updated with task, agent, and CI results.
5. Operators inspect the Twin summary and iterate with another explicit request.

## Related Docs
- Setup guide: [SETUP.md](SETUP.md)
- Unity MLOps scaffold setup: [UNITY_MLOPS_SETUP.md](UNITY_MLOPS_SETUP.md)
- PR validation workflow: [.github/workflows/pr_validation.yml](.github/workflows/pr_validation.yml)
- Full twin pipeline workflow: [a2a_twin_pipeline.yml](a2a_twin_pipeline.yml)
