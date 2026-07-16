#!/usr/bin/env python
"""Simple script to run the Flask server standalone."""

import sys
import io

# Fix Windows UTF-8 console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from core.server import create_flask_app

if __name__ == "__main__":
    app = create_flask_app()
    print("Starting HealthPipe AI Flask server on http://localhost:8501")
    print("Press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=8501, debug=True, use_reloader=False)
