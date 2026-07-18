#!/usr/bin/env python3
"""Static server for the nnLive probes with cross-origin isolation (COOP/COEP) so ORT-Web
wasm threads work, plus HTTP range support for the large model files. Serves this directory."""
import http.server, socketserver, os, sys

DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8799

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=DIR, **k)
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n)
        name = os.path.basename(self.path.strip('/')) or 'post'
        with open(os.path.join(DIR, name + '.json'), 'wb') as fh:
            fh.write(body)
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

socketserver.TCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("", PORT), H) as httpd:
    print(f"serving {DIR} at http://localhost:{PORT}  (COOP/COEP on)")
    httpd.serve_forever()
