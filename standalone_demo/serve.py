#!/usr/bin/env python3
"""
Simple HTTP server for the Gemma 3n MediaPipe demo
Designed to work with ngrok for HTTPS tunneling
"""

import http.server
import os
import socketserver
from pathlib import Path
import time
from datetime import datetime, timedelta
from email.utils import formatdate

PORT = 8081

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Add security headers for WebGPU (required for HTTPS context via ngrok)
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        
        # Add cache headers for .task files to prevent unnecessary downloads
        if self.path.endswith('.task'):
            # Set Cache-Control header with 24-hour max-age and immutable flag
            self.send_header('Cache-Control', 'public, max-age=86400, immutable')
            
            # Set Expires header to 24 hours in the future
            future_date = datetime.now() + timedelta(hours=24)
            self.send_header('Expires', formatdate(time.mktime(future_date.timetuple()), localtime=False, usegmt=True))
            
            print(f"Added cache headers for .task file: {self.path}")
            
        super().end_headers()

def create_server():
    # Change to the current directory
    os.chdir(Path(__file__).parent)
    
    # Create the server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"Serving HTTP on port {PORT}")
        print(f"Local access: http://localhost:{PORT}/demo.html")
        print("\nTo expose via ngrok:")
        print(f"ngrok http --url=gemma3n.ngrok.app {PORT}")
        print("Then access: https://gemma3n.ngrok.app/demo.html")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    create_server()