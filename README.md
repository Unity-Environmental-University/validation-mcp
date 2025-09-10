# validation-mcp (suite module)

This is the externalized Validation MCP server migrated from lisa_brain/validation_mcp.

- Entrypoint (stdio): `python -m validation_mcp.mcp_server`
- CLI: `validation-mcp` (via pyproject scripts)
- Kanban integration via `kanban-mcp` if configured
- Optional sibling dependency: `story-goal-mcp` for acceptance criteria DB

Environment variables:
- KANBAN_MCP_HTTP / KANBAN_MCP_CMD / KANBAN_MCP_TOKEN
- VALIDATION_MOVE_ON_PASS / VALIDATION_MOVE_ON_FAIL
- VALIDATION_USER_KEY
- OPENAI_API_KEY or OLLAMA_HOST/OLLAMA_MODEL for grading tools

Suite layout assumptions:
- Sibling `story-goal-mcp` provides `story_goals.db` for criteria lookup.
- If absent, criteria must be supplied directly to `validate_story`.

