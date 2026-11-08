"""Human-only replay server. Explicitly denies private banks and grader sidecars."""
import argparse,mimetypes,re
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8815);a=p.parse_args();root=Path(a.root).resolve()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  name=unquote(urlsplit(self.path).path).lstrip('/') or 'report/index.html'
  parts=Path(name).parts
  allowed=(len(parts)==2 and parts[0]=='report' and parts[1] in ('index.html','data.json','curriculum.png','curriculum.pdf','rejections.png'))
  allowed=allowed or (len(parts)==4 and parts[0] in ('gpt-study','gpt-private-continuation','gpt-private-continuation-2','gpt-native-adaptive') and parts[1]=='trials' and re.fullmatch(r'[A-Za-z0-9_-]+',parts[2]) and re.fullmatch(r'\d{3}\.png',parts[3]))
  path=(root/name).resolve()
  if not allowed or not path.is_relative_to(root) or not path.is_file():self.send_error(404);return
  data=path.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
