#!/bin/sh
# Idempotent: register audiorise MCP server in opencode (~/.config/opencode/opencode.json).
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python3}"
CFG="$HOME/.config/opencode/opencode.json"
mkdir -p "$(dirname "$CFG")"
"$PY" - "$ROOT" "$CFG" <<'EOF'
import json, sys
root, cfg = sys.argv[1], sys.argv[2]
try:
    with open(cfg) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
srv = {
    "type": "local",
    "command": [sys.executable, "-m", "audio_mcp"],
    "environment": {"AUDIORISE_MCP_ROOTS": root + ":" + "/tmp"},
}
data.setdefault("mcp", {})["audiorise"] = srv
with open(cfg, "w") as f:
    json.dump(data, f, indent=2)
print(f"audiorise MCP registered in {cfg} (cwd={root})")
EOF
