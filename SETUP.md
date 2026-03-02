# A2A Digital Twin - Setup Guide

## System identity and constitutional model
This file is the operational onboarding guide. The canonical system identity and governance model lives in [README.md](README.md#what-is-a2a_mcp), which defines A2A_MCP in this repository as a deterministic, governed control plane with auditable state and artifact lineage surfaces.

## Step 1: Clone your repo and add the extension files

```bash
git clone https://github.com/adaptco-main/AgentADK
cd AgentADK

# Copy all files from this scaffold into the repo root (if using the scaffold tarball)
cp -r a2a-digital-twin/* .
```

Legacy note: Some prompts and examples may still reference the prior `A2A_MCP` naming.

## Step 2: Set environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

## Step 3: Build the RAG embedding store

```bash
# Installs into existing venv
pip install numpy openai httpx fastmcp pytest-json-report

# Build the vertical tensor slice (requires OPENAI_API_KEY)
python bootstrap_digital_twin.py --build-rag

# Verify: should print top-k results
python rag/vertical_tensor_slice.py --query "how does IntentEngine route tasks"
```

## Step 4: Create Airtable Base

1. Go to `airtable.com` and create a base named `A2A Digital Twin`.
2. Create these 4 tables.

### Table: Tasks

| Field Name | Type | Notes |
| --- | --- | --- |
| Name | Single line | Primary field |
| Status | Single select | Backlog, Ready, In Progress, In Review, Done, Blocked |
| Agent Role | Single select | managing_agent, orchestration_agent, architecture_agent, coder, tester, researcher, judge, digital_twin |
| Workflow Stage | Single select | 1-Intake, 2-Research, 3-Architect, 4-Implement, 5-Verify, 6-Checkpoint, 7-Deploy |
| Description | Long text | Full task description |
| Acceptance Criteria | Long text | One criterion per line |
| Browser Steps | Long text | One Playwright action per line |
| GitHub Action | Single line | Example: `ci.yml` or `a2a_twin_pipeline.yml` |
| Office Checkpoint | Single select | (empty), word, excel, outlook, all |
| Related Tasks | Link to Tasks | Self-referential dependencies |

### Table: Roles

| Field Name | Type | Notes |
| --- | --- | --- |
| Name | Single line | Primary field |
| Agent Class | Single line | Python class name from `agents/` |
| System Prompt | Long text | Full system prompt for this role |
| Tools | Long text | Comma-separated tool names |
| MCP Tools | Long text | Comma-separated MCP tool names |

### Table: Workflows

| Field Name | Type | Notes |
| --- | --- | --- |
| Name | Single line | Primary field |
| Stages | Multiple select | Same values as Task Workflow Stage |
| Tasks | Link to Tasks | All tasks in this workflow |
| Trigger | Single select | manual, push, schedule, webhook |
| GitHub Action File | Single line | Example: `a2a_twin_pipeline.yml` |

### Table: Actions (for GitHub Actions tracking)

| Field Name | Type | Notes |
| --- | --- | --- |
| Name | Single line | Example: `CI Run - abc1234` |
| Run ID | Single line | GitHub Actions run ID |
| Status | Single select | success, failure, in_progress |
| Triggered By | Link to Tasks | Which task triggered this run |
| Run URL | URL | Link to GitHub Actions run |
| Timestamp | Date | When the run completed |

## Step 5: Add GitHub Secrets

In your GitHub repository settings, add these as Actions secrets:

```text
OPENAI_API_KEY          - for embeddings
AIRTABLE_API_KEY        - from airtable.com/account
AIRTABLE_BASE_ID        - from airtable URL: airtable.com/appXXXXXXXX/...
OFFICE_TENANT_ID        - from Azure AD App Registration
OFFICE_CLIENT_ID        - from Azure AD App Registration
OFFICE_CLIENT_SECRET    - from Azure AD App Registration
OFFICE_USER_EMAIL       - Office 365 user to file docs under
HANDOFF_EMAIL           - where to send handoff emails
PERPLEXITY_API_KEY      - from perplexity.ai/api
```

## Step 6: Add the GitHub Actions workflow

```bash
cp a2a_twin_pipeline.yml .github/workflows/
git add .github/workflows/a2a_twin_pipeline.yml
git commit -m "feat: add A2A digital twin pipeline"
git push
```

## Commit Message Template (Recommended)

This repository includes a reusable commit template at `.gitmessage` based on Conventional Commits.

### One-time local setup

Run from the repository root:

```bash
git config commit.template .gitmessage
git config --get commit.template
```

Expected output:

```text
.gitmessage
```

### Commit grammar

Use this first-line format:

```text
<type>(<scope>): <subject>
```

Allowed `type` values:
`feat`, `fix`, `ci`, `build`, `refactor`, `test`, `docs`, `chore`, `perf`, `revert`

Subject rules:
- Use imperative mood (`add`, `fix`, `update`)
- No trailing period
- Keep it at 72 characters or less when possible

Optional body/footer fields in the template:
- `Why`
- `What changed`
- `Validation`
- `Refs`
- `BREAKING CHANGE`

### Usage behavior

- `git commit` opens `.gitmessage` in your editor.
- `git commit -m "..."` bypasses the template.

### Copy/paste examples

```bash
git commit -m "feat(rag): add deterministic reranking for retrieval"
git commit -m "fix(webhooks): handle missing task_id in dispatch payload"
git commit -m "ci(github-actions): split lint and tests into separate jobs"
```

### Troubleshooting

- If the template does not appear, confirm local config with `git config --local --get commit.template`.
- Confirm you are committing from this repository and not a different checkout.
- Verify your editor is configured for Git commit messages and did not abort on empty content.
- To test template rendering directly, run `git commit --allow-empty` and inspect the opened draft.

## Step 7: Start the MCP Server

```bash
# Stdio mode (for Claude Desktop / Cursor / VS Code MCP extension)
python bootstrap_digital_twin.py

# HTTP mode (for remote agents)
MCP_TRANSPORT=http MCP_PORT=8080 python bootstrap_digital_twin.py
```

## Step 8: Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "a2a-digital-twin": {
      "command": "python",
      "args": ["/path/to/AgentADK/bootstrap_digital_twin.py"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "AIRTABLE_API_KEY": "pat...",
        "AIRTABLE_BASE_ID": "app...",
        "PERPLEXITY_API_KEY": "pplx-..."
      }
    }
  }
}
```

Any LLM that speaks MCP (Claude, Cursor, VS Code Copilot, and others) can then call:
- `search_repo` for semantic repository search
- `spawn_agent` to dispatch an A2A subagent by task description
- `get_twin_state` to inspect task, agent, and CI state
- `run_tests` to execute pytest and collect results
- `git_commit` to commit and push through MCP tooling

---

## Mental Model: Execution Flow

```text
User request
  -> bootstrap_digital_twin.py (MCP server)
  -> spawn_agent(task)
  -> agent tools (search_repo, search_web, read/write, run_tests, git_commit)
  -> GitHub Actions workflow (a2a_twin_pipeline.yml)
  -> Twin state + CI status update
  -> next explicit user request
```
