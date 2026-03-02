# A2A Digital Twin - Setup Guide

## System identity and constitutional model

This file is the operational setup guide. The canonical system identity
and governance model is in
[README.md](README.md#what-is-a2a_mcp).

## Step 1: Clone your repo and add extension files

```bash
git clone https://github.com/adaptco-main/AgentADK
cd AgentADK

# Optional: copy scaffold files into this repo root
cp -r a2a-digital-twin/* .
```

Legacy note: older prompts may still reference the name `A2A_MCP`.

## Step 2: Set environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

## Step 3: Build the RAG embedding store

```bash
pip install numpy openai httpx fastmcp pytest-json-report
python bootstrap_digital_twin.py --build-rag
python rag/vertical_tensor_slice.py --query "how does IntentEngine route tasks"
```

## Step 4: Create Airtable base

Create a base named `A2A Digital Twin` with these tables:

1. `Tasks`
2. `Roles`
3. `Workflows`
4. `Actions`

### Tasks table fields

- `Name` (single line, primary)
- `Status` (single select)
- `Agent Role` (single select)
- `Workflow Stage` (single select)
- `Description` (long text)
- `Acceptance Criteria` (long text)
- `Browser Steps` (long text)
- `GitHub Action` (single line)
- `Office Checkpoint` (single select)
- `Related Tasks` (link to Tasks)

### Roles table fields

- `Name` (single line, primary)
- `Agent Class` (single line)
- `System Prompt` (long text)
- `Tools` (long text)
- `MCP Tools` (long text)

### Workflows table fields

- `Name` (single line, primary)
- `Stages` (multiple select)
- `Tasks` (link to Tasks)
- `Trigger` (single select)
- `GitHub Action File` (single line)

### Actions table fields

- `Name` (single line)
- `Run ID` (single line)
- `Status` (single select)
- `Triggered By` (link to Tasks)
- `Run URL` (URL)
- `Timestamp` (date)

## Step 5: Add GitHub secrets

Add these repository Action secrets:

```text
OPENAI_API_KEY
AIRTABLE_API_KEY
AIRTABLE_BASE_ID
OFFICE_TENANT_ID
OFFICE_CLIENT_ID
OFFICE_CLIENT_SECRET
OFFICE_USER_EMAIL
HANDOFF_EMAIL
PERPLEXITY_API_KEY
```

## Step 6: Add the GitHub Actions workflow

```bash
cp a2a_twin_pipeline.yml .github/workflows/
git add .github/workflows/a2a_twin_pipeline.yml
git commit -m "feat: add A2A digital twin pipeline"
git push
```

## Commit Message Template (recommended)

This repository includes `.gitmessage` with Conventional Commits.

### One-time local setup

```bash
git config commit.template .gitmessage
git config --get commit.template
```

### Commit grammar

Use:

```text
<type>(<scope>): <subject>
```

Allowed `type` values:
`feat`, `fix`, `ci`, `build`, `refactor`, `test`, `docs`,
`chore`, `perf`, `revert`

Subject rules:

- use imperative mood
- no trailing period
- keep it near 72 chars

Optional body/footer fields:

- `Why`
- `What changed`
- `Validation`
- `Refs`
- `BREAKING CHANGE`

## Step 7: Start the MCP server

```bash
# Stdio mode (Claude Desktop / Cursor / VS Code MCP extension)
python bootstrap_digital_twin.py

# HTTP mode
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

Once connected, MCP-capable clients can call:

- `search_repo`
- `spawn_agent`
- `get_twin_state`
- `run_tests`
- `git_commit`

---

## Mental Model: Execution Flow

```text
user request
  -> bootstrap_digital_twin.py
  -> spawn_agent(task)
  -> tool calls (search, read/write, tests, commit)
  -> GitHub Actions (a2a_twin_pipeline.yml)
  -> Twin state + CI status update
  -> next explicit user request
```
