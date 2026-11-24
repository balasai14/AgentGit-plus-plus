"""Simple local server to preview the AgentGit++ Dashboard.

Usage:
    python web/serve.py [--port 8000]
"""
import argparse
import http.server
import os
import socketserver
import sys
import webbrowser

PORT = 8000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}"
        print("=" * 60)
        print(f"AgentGit++ Dashboard server active at: {url}")
        print("Press Ctrl+C to stop.")
        print("=" * 60)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()

