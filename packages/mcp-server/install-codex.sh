#!/bin/sh
# Idempotent: register audiorise MCP server in codex (~/.codex/config.toml).
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python3}"
CFG="$HOME/.codex/config.toml"
mkdir -p "$(dirname "$CFG")"
touch "$CFG"
if grep -q 'mcp_servers.audiorise' "$CFG"; then
  echo "audiorise already in $CFG"
  exit 0
fi
cat >>"$CFG" <<EOF

[mcp_servers.audiorise]
command = "$PY"
args = ["-m", "audio_mcp"]
cwd = "$ROOT"
env = { "AUDIORISE_MCP_ROOTS" = "$ROOT:/tmp" }
EOF
echo "audiorise MCP appended to $CFG"
