# Implementation Plan

[Overview]
This implementation remediates markdownlint issues across all git-tracked Markdown documentation files in the repository while preserving document meaning.

The immediate workspace diagnostics identify rule violations in `SETUP.md` (MD060 table-column-style, MD040 fenced-code-language, and MD032 blanks-around-lists). The requested scope is broader than a single file, so implementation should treat these diagnostics as the starting point for a full repository Markdown lint pass on tracked `.md` files: `SETUP.md`, `UNITY_MLOPS_SETUP.md`, `MERGE_RECEIPT_RESPONSE.md`, and `matrix_dash/README.md`.

The high-level approach is documentation-only remediation: normalize table formatting to the configured aligned style, add explicit code-fence language identifiers to unlabeled fenced blocks, and enforce blank-line spacing around list blocks where required. No product runtime behavior should change. Validation should be done by rerunning markdownlint on the same file set and confirming the previously reported warnings are resolved with no regressions.

[Types]
No runtime application type-system changes are required, but the implementation will use a structured lint-remediation data model for deterministic execution and verification.

Proposed planning data structures (conceptual, for implementation workflow and acceptance tracking):

- `MarkdownLintIssue`
  - `file_path: str` (relative repository path; must resolve to a tracked `.md` file)
  - `line: int` (1-based line number; must be > 0)
  - `rule_id: str` (e.g., `MD032`, `MD040`, `MD060`)
  - `message: str` (exact lint diagnostic text)
  - `status: Literal["open", "fixed", "verified"]`
  - Validation rules:
    - `rule_id` must match markdownlint rule token format `MD\d{3}`.
    - `status` transitions only `open -> fixed -> verified`.

- `TableAlignmentSpec`
  - `file_path: str`
  - `section_heading: str`
  - `column_count: int` (must be consistent across header, delimiter, and all rows)
  - `style: Literal["aligned"]`
  - Validation rules:
    - Every row in a table block must contain identical pipe-delimited column count.
    - Pipe positions must align to header column boundaries for aligned style.

- `FenceLanguageSpec`
  - `file_path: str`
  - `start_line: int`
  - `language: Literal["bash", "json", "text", "powershell"]`
  - Validation rules:
    - Fenced blocks may not be language-empty.
    - Chosen language must reflect block content semantics.

[Files]
The remediation primarily modifies Markdown documentation files, with `SETUP.md` as the main change surface and repository-wide lint verification across all tracked `.md` files.

Detailed breakdown:

- New files to be created:
  - `implementation_plan.md` — authoritative planning artifact describing scope, technical decisions, and execution order.

- Existing files to be modified:
  - `SETUP.md`
    - Reformat all Airtable schema tables under Steps 4/5 sections so table pipes conform to MD060 `aligned` style.
    - Add explicit fence language to currently unlabeled fenced blocks (notably secrets list and mental-model flow block) to satisfy MD040.
    - Insert required blank lines around list groups in commit-grammar and capabilities sections to satisfy MD032.
    - Preserve all existing instructional content and command semantics.
  - `UNITY_MLOPS_SETUP.md` (conditional)
    - Apply fixes only if repo-wide markdownlint reports violations.
  - `MERGE_RECEIPT_RESPONSE.md` (conditional)
    - Apply fixes only if repo-wide markdownlint reports violations.
  - `matrix_dash/README.md` (conditional)
    - Apply fixes only if repo-wide markdownlint reports violations.

- Files to be deleted or moved:
  - None.

- Configuration file updates:
  - No mandatory config file additions.
  - Optional (only if needed for deterministic CI/local parity): add markdownlint config in a follow-up change; do not disable failing rules as a shortcut for remediation.

[Functions]
No executable function logic is being introduced or changed because this is a documentation-format remediation task.

Detailed breakdown:

- New functions:
  - None.

- Modified functions:
  - None.

- Removed functions:
  - None.

[Classes]
No class definitions are introduced, modified, or removed in this documentation-only remediation.

Detailed breakdown:

- New classes:
  - None.

- Modified classes:
  - None.

- Removed classes:
  - None.

[Dependencies]
No persistent runtime or test dependency changes are required for this implementation.

Details:

- `requirements.txt` remains unchanged.
- Markdown lint verification should use existing local tooling (`npx`) without introducing committed Node package manifests unless explicitly requested later.
- If ephemeral `npx` execution is unavailable in a constrained environment, fallback validation can be performed via VS Code markdownlint diagnostics plus targeted rule checks.

[Testing]
Testing will use lint-based validation to confirm all markdown diagnostics are resolved and no new markdownlint issues are introduced.

Test requirements and validation strategy:

- Baseline scan (before edits) across tracked Markdown files.
- Post-change scan across tracked Markdown files with the same rule set/environment.
- Focused verification that prior diagnostics are gone:
  - MD060 table-column-style warnings in `SETUP.md`
  - MD040 fenced-code-language warnings in `SETUP.md`
  - MD032 blanks-around-lists warnings in `SETUP.md`
- Optional spot-check rendering in VS Code Markdown preview to ensure readability after table alignment changes.
- Acceptance criteria:
  - Zero markdownlint warnings on tracked Markdown files within scope.
  - No semantic changes to setup instructions.

[Implementation Order]
The implementation sequence first establishes a repository-wide lint baseline, then applies deterministic fixes in `SETUP.md`, then validates and addresses any remaining Markdown warnings across the repo.

1. Enumerate tracked Markdown files and run a baseline markdownlint scan for full-scope visibility.
2. Update `SETUP.md` table blocks to satisfy MD060 aligned column style.
3. Update `SETUP.md` unlabeled code fences with explicit languages to satisfy MD040.
4. Add blank-line separators around list blocks in `SETUP.md` to satisfy MD032.
5. Re-run markdownlint across all tracked `.md` files and capture remaining warnings, if any.
6. Apply minimal, content-preserving fixes to any additional Markdown files flagged by step 5.
7. Perform final markdownlint pass and confirm zero warnings in scope.
8. Prepare implementation summary with changed-file list and verification outputs.
