# AudioRise MCP server

`audiorise` MCP server (official SDK v2, stdio) with four tools —
`probe_audio`, `separate_audio`, `label_stems`, `summarize_stems` —
returning structured output. Paths are jailed to `AUDIORISE_MCP_ROOTS`
(default: repo root + tmp).

```bash
pip install -e ../core -e .            # from packages/mcp-server/
python -m audio_mcp                    # stdio
audiorise-mcp                          # same, via console script
```

One-command client setup (idempotent, machine-local paths auto-detected):

```bash
sh packages/mcp-server/install-opencode.sh   # → ~/.config/opencode/opencode.json
sh packages/mcp-server/install-codex.sh     # → ~/.codex/config.toml
sh packages/mcp-server/install-claude.sh     # → claude mcp add (user scope)
```

Env: `AUDIORISE_MCP_ROOTS` (`:`-separated, `;` on Windows). `separate_audio`
needs `pip install -e ../core[separation]` and runs Demucs on CPU — slow for
full songs; `probe_audio` / `summarize_stems` stay light (ffprobe only).
