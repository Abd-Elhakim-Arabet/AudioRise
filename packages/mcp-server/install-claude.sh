#!/bin/sh
# Register audiorise MCP server with Claude Code (user scope).
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python3}"
claude mcp add audiorise --scope user --env "AUDIORISE_MCP_ROOTS=$ROOT:/tmp" -- "$PY" -m audio_mcp
