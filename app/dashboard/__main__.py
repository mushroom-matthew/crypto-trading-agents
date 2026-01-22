"""CLI entry point for running the dashboard server."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.dashboard.server:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
    )

