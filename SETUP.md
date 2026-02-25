# A2A Digital Twin — Setup Guide

## Step 1: Clone your repo and add the extension files

```bash
git clone https://github.com/adaptco-main/A2A_MCP
cd A2A_MCP

# Copy all files from this scaffold into the repo root
cp -r a2a-digital-twin/* .
```

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

1. Go to airtable.com → Create Base → Name: **"A2A Digital Twin"**
2. Create these 4 tables:

### Table: Tasks

| Field Name          | Type            | Notes                               |
|---------------------|-----------------|-------------------------------------|
| Name                | Single line     | Primary field                       |
| Status              | Single select   | Backlog, Ready, In Progress, In Review, Done, Blocked |
| Agent Role          | Single select   | managing_agent, orchestration_agent, architecture_agent, coder, tester, researcher, judge, digital_twin |
| Workflow Stage      | Single select   | 1-Intake, 2-Research, 3-Architect, 4-Implement, 5-Verify, 6-Checkpoint, 7-Deploy |
| Description         | Long text       | Full task description                |
| Acceptance Criteria | Long text       | One criterion per line               |
| Browser Steps       | Long text       | One Playwright action per line       |
| GitHub Action       | Single line     | e.g. "ci.yml" or "a2a_twin_pipeline.yml" |
| Office Checkpoint   | Single select   | (empty), word, excel, outlook, all  |
| Related Tasks       | Link to Tasks   | Self-referential for dependencies    |

### Table: Roles

| Field Name     | Type        | Notes                                    |
|----------------|-------------|------------------------------------------|
| Name           | Single line | Primary field                             |
| Agent Class    | Single line | Python class name from agents/            |
| System Prompt  | Long text   | Full system prompt for this role          |
| Tools          | Long text   | Comma-separated tool names               |
| MCP Tools      | Long text   | Comma-separated MCP tool names           |

### Table: Workflows

| Field Name         | Type            | Notes                              |
|--------------------|-----------------|------------------------------------|
| Name               | Single line     | Primary field                      |
| Stages             | Multiple select | Same values as Task Workflow Stage |
| Tasks              | Link to Tasks   | All tasks in this workflow         |
| Trigger            | Single select   | manual, push, schedule, webhook    |
| GitHub Action File | Single line     | e.g. "a2a_twin_pipeline.yml"       |

### Table: Actions (for GitHub Actions tracking)

| Field Name    | Type        | Notes                            |
|---------------|-------------|----------------------------------|
| Name          | Single line | e.g. "CI Run — abc1234"          |
| Run ID        | Single line | GitHub Actions run ID             |
| Status        | Single select | success, failure, in_progress   |
| Triggered By  | Link to Tasks | Which task triggered this run   |
| Run URL       | URL         | Link to GitHub Actions run       |
| Timestamp     | Date        | When the run completed           |

## Step 5: Add GitHub Secrets

In your GitHub repo → Settings → Secrets → Actions:

```
OPENAI_API_KEY          — for embeddings
AIRTABLE_API_KEY        — from airtable.com/account
AIRTABLE_BASE_ID        — from airtable URL: airtable.com/appXXXXXXXX/...
OFFICE_TENANT_ID        — from Azure AD App Registration
OFFICE_CLIENT_ID        — from Azure AD App Registration
OFFICE_CLIENT_SECRET    — from Azure AD App Registration
OFFICE_USER_EMAIL       — Office 365 user to file docs under
HANDOFF_EMAIL           — where to send handoff emails
PERPLEXITY_API_KEY      — from perplexity.ai/api
```

## Step 6: Add the GitHub Actions workflow

```bash
cp integrations/github/a2a_twin_pipeline.yml .github/workflows/
git add .github/workflows/a2a_twin_pipeline.yml
git commit -m "feat: add A2A digital twin pipeline"
git push
```

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
      "args": ["/path/to/A2A_MCP/bootstrap_digital_twin.py"],
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

Now any LLM that speaks MCP (Claude, Cursor, VS Code Copilot, etc.) can:
- `search_repo` — NDP search over your entire codebase
- `spawn_agent` — Spawn any A2A subagent by task description
- `get_twin_state` — See live task/agent/CI status
- `run_tests` — Run pytest and see results
- `git_commit` — Commit and push from the LLM

---

## Mental Model: How a long-horizon task flows

```
You type: "Implement the Postgres ArtifactStore replacing InMemoryArtifactStore"
    ↓
bootstrap_digital_twin.py (MCP server)
    ↓
spawn_agent("Implement Postgres ArtifactStore")
    ↓
NDP routes to "coder" (highest dot product with task embedding)
    ↓
CoderAgent:
  1. search_repo("ArtifactStore interface")  → finds artifact-store/in-memory-store.ts
  2. search_web("SQLAlchemy pgvector cosine") → Perplexity fills knowledge gap
  3. read_file("schemas/agent_artifacts.py") → understands MCPArtifact schema
  4. write_file("agents/pg_artifact_store.py", <new code>)
  5. run_tests("test_artifact_store")         → verify
  6. git_commit("feat: Postgres artifact store", push=True)
    ↓
GitHub Actions picks up push → a2a_twin_pipeline.yml
    ↓
  Job 1: Rebuild embeddings (new file added)
  Job 2: Sync Airtable task status → In Review
  Job 3: Run full test suite + verify fossil chain
  Job 4: Write Word handoff doc + Excel metrics + send Outlook email
  Job 5: Mark Airtable task → Done
    ↓
Digital Twin state: task complete, CI green, handoff doc in OneDrive
```
