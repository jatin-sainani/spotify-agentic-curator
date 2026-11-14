"""Launcher for MCP clients: `<venv python> /abs/path/to/mcp_server.py`.

MCP clients start servers from their own working directory, where `python -m app`
can't import this project. This script adds the repo to sys.path first, so a
client config only needs absolute paths (see README "Use it from other agents").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.mcp_server import serve  # noqa: E402

if __name__ == "__main__":
    serve()
