---
name: mcp-maker
description: >
  Scaffold, register, and test a new MCP server for Claude Code. Use when a task
  needs structured access to an external API (REST, database, file system) that
  isn't covered by an existing plugin. Triggers on: "make an MCP", "I need to
  query [API name]", "create a tool for [service]", "connect to [database/API]",
  or when you determine that repeated API calls in a session would benefit from
  a dedicated tool.
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# MCP Maker

Create lightweight, single-file MCP servers using Python FastMCP and register them
with Claude Code — all within the current session.

## When to use this skill

- The user (or you) needs to repeatedly query an external API during a task.
- A premade MCP plugin doesn't exist for the target service.
- You need structured tool access (typed inputs/outputs) rather than raw `curl`.

## When NOT to use this skill

- An official Anthropic plugin already covers the service (check `/plugin list` first).
- The API will only be called once — just use `curl` or `requests` directly.
- The task is read-only exploration — a quick script may be simpler than an MCP.

## Constraints (from CLAUDE.md)

- **Transport**: Always use `stdio`. No SSE/HTTP unless the user explicitly requests remote access.
- **Credentials**: Never hardcode API keys. Use `os.environ.get("KEY_NAME")` and document required env vars.
- **Dependencies**: Minimize. FastMCP + the API client library, nothing more.
- **Memory**: This runs on an M1 Pro with 16 GB. The server must be lightweight.
- **Timeout**: Wrap all network calls in a timeout (default 30s) so a bad endpoint doesn't hang the session.

## Workflow

### Step 1: Gather requirements

Before writing code, confirm with the user:
1. **Target API/service** — What are we connecting to?
2. **Tools needed** — What operations? (e.g., "search genes by symbol", "fetch expression data")
3. **Auth method** — API key? OAuth? None? Where is the key stored?
4. **Env var names** — What should the environment variable(s) be called?

### Step 2: Scaffold the server

Use the template at `templates/server_template.py` as a starting point. Create the server file at:

```
.claude/mcp-servers/<server-name>.py
```

Naming convention: lowercase, hyphens, descriptive. E.g., `ensembl-genes.py`, `geo-metadata.py`.

Each `@mcp.tool()` function must have:
- A clear, specific docstring (Claude uses this to decide when to call the tool).
- Type hints on all parameters and return type.
- Error handling that returns a useful message, not a traceback.
- A timeout on network calls.

### Step 3: Test the server

Run a smoke test before registering:

```bash
# Verify it starts without errors
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}},"id":1}' | uv run --with fastmcp python .claude/mcp-servers/<server-name>.py
```

If the server responds with a valid JSON-RPC response, it's working.

### Step 4: Register with Claude Code

```bash
claude mcp add <server-name> --scope project -- \
  uv run --with fastmcp python .claude/mcp-servers/<server-name>.py
```

If the server needs additional pip packages (e.g., `requests`, `biopython`):

```bash
claude mcp add <server-name> --scope project -- \
  uv run --with fastmcp --with requests python .claude/mcp-servers/<server-name>.py
```

If env vars are needed:

```bash
claude mcp add <server-name> --scope project \
  -e API_KEY="${API_KEY}" -- \
  uv run --with fastmcp python .claude/mcp-servers/<server-name>.py
```

### Step 5: Verify registration

Tell the user to run `/mcp` to confirm the server appears and is connected.

### Step 6: Document

Add a brief entry to the project's `CONTEXT.md` or `README.md`:
- Server name and purpose
- Required env vars
- Tools exposed
- Whether it's disposable or should be committed to git

## Cleanup

When a server is no longer needed:

```bash
claude mcp remove <server-name>
```

And optionally delete the file:

```bash
rm .claude/mcp-servers/<server-name>.py
```

## Bioinformatics-specific patterns

Common MCP servers you might create for this domain:

| Server | API | Typical tools |
|--------|-----|---------------|
| `ensembl-genes` | Ensembl REST | `lookup_gene`, `get_sequence`, `get_homologs` |
| `geo-metadata` | NCBI GEO E-utils | `search_geo`, `get_series_info`, `list_samples` |
| `uniprot-lookup` | UniProt REST | `search_protein`, `get_entry`, `get_features` |
| `chembl-compounds` | ChEMBL REST | `search_compound`, `get_bioactivities` |
| `opentargets` | Open Targets GraphQL | `disease_associations`, `drug_mechanisms` |

These are suggestions — build only what the current task requires.
