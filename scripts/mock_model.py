"""Local wiring smoke only: always submit, no model, no repair skill."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        model=body.get('model','scripted-submit-smoke')
        content='not valid JSON' if model=='scripted-malformed-smoke' else json.dumps({
            'action_json':json.dumps({'type':'submit'}),
            'summary':'Scripted wiring test: unchanged submission; not a model repair.'})
        response={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':content}}]}
        wire=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8008);a=p.parse_args()
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
