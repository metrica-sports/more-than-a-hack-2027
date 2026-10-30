"""Launcher for Barca Vision Labeler.

Initializes the index, starts the local Flask server on http://127.0.0.1:5000,
and opens the default web browser automatically.
"""

import os
import sys
import time
import webbrowser
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from app import app, initialize  # noqa: E402


def open_browser():
    """Wait for server boot and launch browser."""
    time.sleep(1.2)
    url = "http://127.0.0.1:5000"
    print(f"[Labeler] Opening browser at {url} ...")
    webbrowser.open(url)


def main():
    print("=" * 60)
    print("⚽  BARCA VISION LABELER -- TACTICAL ANNOTATION STUDIO")
    print("=" * 60)
    print("Building high-performance frame index...")
    initialize()
    print("Ready! Starting web server on http://127.0.0.1:5000")
    print("Press Ctrl+C in terminal to stop.")
    print("=" * 60)

    # Launch browser thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run server
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
