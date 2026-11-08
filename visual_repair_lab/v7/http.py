"""Loopback operator GUI and capability-scoped actor endpoints."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .server import Store,Conflict
from .benchmark import Benchmarks


def application(store,runs):
    benchmarks=Benchmarks(store,runs)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def send(self,status,result,content_type='application/json'):
            data=result.encode() if isinstance(result,str) else json.dumps(result,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            path=urlparse(self.path)
            if path.path=='/':
                self.send(200,Path(__file__).with_name('gui.html').read_text(),'text/html; charset=utf-8')
            elif path.path=='/health':self.send(200,{'ok':True,'version':'7.0.0','runtime_bank_seal':store.seal})
            elif path.path=='/api/tasks':
                self.send(200,{'ok':True,'tasks':[{'id':t['id'],'family':t['family'],'budget':t['budget'],'split':t['split']}
                    for t in store.bank.values()]})
            elif path.path=='/api/runs':
                self.send(200,{'ok':True,'runs':[p.name for p in benchmarks.directory.iterdir() if p.is_dir() and (p/'config.json').exists()]})
            elif path.path=='/api/run':
                try:self.send(200,benchmarks.summary(parse_qs(path.query)['id'][0]))
                except (ValueError,KeyError,FileNotFoundError):self.send(404,{'ok':False,'error':'unknown_run'})
            else:self.send(404,{'ok':False,'error':'unknown_endpoint'})

        def do_POST(self):
            try:
                self.connection.settimeout(15)
                origin=self.headers.get('Origin')
                if origin and origin!='http://'+self.headers.get('Host',''):raise ValueError('Cross-origin request')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=100000:raise ValueError('Request size')
                body=json.loads(self.rfile.read(size))
                if self.path=='/episodes':result=store.create(body['task_id'],body.get('evidence','image'),body.get('mode','evaluation'))
                elif self.path=='/resume':result=store.get(body['episode_id'])
                elif self.path=='/step':result=store.step(body['episode_id'],body['request_id'],body['expected_step'],body['action'])
                elif self.path=='/api/benchmark':result=benchmarks.start(body)
                elif self.path=='/api/resume-benchmark':
                    benchmarks.launch(body['run_id'],body.get('key',''));result={'ok':True,'run_id':body['run_id']}
                else:raise ValueError('Unknown endpoint')
                self.send(200 if result.get('ok') else 503,result)
            except Conflict:
                self.send(409,{'ok':False,'error':'step_conflict','retryable':False,'scored':False})
            except (ValueError,KeyError,TypeError,FileNotFoundError):
                self.send(400,{'ok':False,'error':'invalid_request','scored':False})
            except Exception:
                self.send(503,{'ok':False,'error':'server_interruption','retryable':True,'scored':False})
    return Handler


def serve(database,bank_path,host='127.0.0.1',port=8822,timeout=45,runs=None):
    store=Store(database,json.loads(Path(bank_path).read_text()),timeout)
    class Server(ThreadingHTTPServer):
        request_queue_size=64
        daemon_threads=True
    Server((host,port),application(store,runs or str(Path(database).with_suffix('.runs')))).serve_forever()
