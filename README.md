# AgentADK (A2A Digital Twin)

## What Is A2A_MCP?

A2A_MCP in this repository is a constitutional, deterministic control
plane for agent-assisted software workflows. State transitions and
artifacts are accepted only through explicit, auditable surfaces.

This document is repo-grounded. It describes controls that exist in this
codebase today and marks missing controls as target-state roadmap items.

## Constitutional Invariants Enforced Here

- Deterministic state evolution:
  Twin state is persisted in `digital_twin/twin_state.json` through
  `digital_twin/twin_registry.py`.
- Content-addressed lineage:
  SHA-256 hashing plus Merkle roots/proofs are implemented in
  `panda_guard/merkle.py`.
- Governed execution surfaces:
  MCP tools and orchestration are wired in `bootstrap_digital_twin.py`
  and `mcp_extensions/claude_code_mcp_server.py`.
- Auditability:
  CI workflows publish test and state outcomes through
  `.github/workflows/pr_validation.yml` and `a2a_twin_pipeline.yml`.

## Governance Corridor (Current Implementation)

The corridor in this repository is governance-by-validated-surfaces:

- Requests enter through MCP entrypoints and registered tools.
- Agent actions operate through tool interfaces and repository mutations.
- CI workflows validate syntax, tests, and guard checks.
- Twin state is updated with explicit task, agent, and CI status.

This is governed autonomy, not unrestricted autonomy.

## Anti-Drift and Anti-Corruption Boundaries

- MCP boundary:
  tool registration separates model decisions from direct mutations.
- Air-gap boundary:
  `panda_guard/air_gap.py` scans and can redact sensitive chunk text.
- Integrity boundary:
  Merkle verification detects embedding artifact tampering.
- CI gate boundary:
  workflows enforce repeatable checks before state advancement.

## What Exists vs What Is Target-State

| Implemented in this repo | Not implemented yet / roadmap |
| --- | --- |
| Twin registry (`twin_registry.py`) | `0x_ENVELOPE_V4_ALPHA_STABLE` |
| SHA-256 + Merkle (`merkle.py`) | `LIFT_VECTOR_ALPHA` |
| Air-gap scan/redaction (`air_gap.py`) | `LIFT_VECTOR_BETA` |
| MCP tool gateway (`bootstrap_digital_twin.py`) | `HAMILTONIAN_VALIDATOR` |
| CI governance workflows | `kernel.index.freeze.completed` |
| Twin + Airtable sync pipeline | corridor-wide `session_id` contracts |
| Audit artifacts in CI | RLS-backed governance controls |

## Operator Mental Model

1. A user request enters through the MCP control surface.
2. Agents execute through registered tools and repo operations.
3. CI validates outputs and guard checks.
4. Twin state is updated with result status.
5. The operator issues the next explicit request.

## Related Docs

- Setup guide: [SETUP.md](SETUP.md)
- Unity scaffold setup: [UNITY_MLOPS_SETUP.md](UNITY_MLOPS_SETUP.md)
- PR validation: [.github/workflows/pr_validation.yml](.github/workflows/pr_validation.yml)
- Full twin pipeline: [a2a_twin_pipeline.yml](a2a_twin_pipeline.yml)
