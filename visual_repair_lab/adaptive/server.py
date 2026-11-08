"""Official actor interface: durable, idempotent, isolated worker evaluations."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .dsl import seal, validate
from .private_checks import validate_private


def runtime_seal():
    import vision_sort
    roots = [Path(__file__).parents[1], Path(vision_sort.__file__).parent]
    return seal({str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for root in roots for p in sorted(root.rglob('*.py'))})


class Store:
    def __init__(self, path, bank, timeout=45):
        self.path, self.timeout = str(path), timeout
        self.bank = {r['spec']['id']: r['spec'] for r in bank['records'] if r['certificate']['accepted']}
        if not self.bank or seal(bank['records']) != bank['bank_sha256']:
            raise ValueError('Invalid bank seal')
        for r in bank['records']:
            validate_private(r['spec']) if r['spec']['family']=='detection_overlay' else validate(r['spec'])
            if r['certificate']['spec_sha256'] != seal(r['spec']):
                raise ValueError('Invalid task certificate binding')
        self.seal = seal({'runtime': runtime_seal(), 'bank': bank['bank_sha256']})
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS metadata(seal TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, config TEXT NOT NULL, actions TEXT NOT NULL, result TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS requests(episode TEXT, request_id TEXT, digest TEXT, result TEXT, PRIMARY KEY(episode,request_id));
              CREATE TABLE IF NOT EXISTS interruptions(episode TEXT, at REAL, kind TEXT);''')
            prior = db.execute('SELECT seal FROM metadata').fetchone()
            if prior and prior[0] != self.seal:
                raise ValueError('Runtime/bank changed: use a new database; old evidence is preserved')
            if not prior:
                db.execute('INSERT INTO metadata VALUES (?)', (self.seal,))
        os.chmod(self.path, 0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=self.timeout+5)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        return db

    def worker(self, config, actions):
        try:
            p = subprocess.run([sys.executable, '-m', 'visual_repair_lab.adaptive.worker'],
                input=json.dumps(dict(config, actions=actions)), capture_output=True, text=True,
                timeout=self.timeout, env={k: v for k, v in os.environ.items()
                                          if k not in ('OPENROUTER_API_KEY', 'OPENAI_API_KEY')})
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': 'evaluation_timeout', 'scored': False, 'retryable': True}
        if p.returncode:
            return {'ok': False, 'error': 'evaluator_interruption', 'scored': False, 'retryable': True}
        try:
            result = json.loads(p.stdout)
            return result if result.get('ok') else {'ok': False, 'error': 'evaluator_interruption', 'scored': False, 'retryable': True}
        except ValueError:
            return {'ok': False, 'error': 'evaluator_protocol', 'scored': False, 'retryable': True}

    def create(self, task_id, evidence='image', mode='evaluation'):
        if task_id not in self.bank or evidence not in ('image', 'numeric_text', 'image_no_footer', 'none') or mode not in ('train', 'evaluation'):
            raise ValueError('Unknown task or condition')
        if mode == 'train' and self.bank[task_id]['split'] != 'train':
            raise ValueError('Dense training feedback is restricted to the train split')
        config = {'spec': self.bank[task_id], 'evidence': evidence, 'mode': mode}
        result = self.worker(config, [])
        if not result['ok']:
            return result
        eid = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute('INSERT INTO episodes VALUES (?,?,?,?)', (eid, json.dumps(config), '[]', json.dumps(result)))
        return dict(result, episode_id=eid)

    def get(self, eid):
        with self.connect() as db:
            row = db.execute('SELECT result FROM episodes WHERE id=?', (eid,)).fetchone()
            if not row:
                raise ValueError('Unknown episode')
            return dict(json.loads(row[0]), episode_id=eid)

    def step(self, eid, request_id, expected_step, action):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100 or type(expected_step) is not int:
            raise ValueError('request_id and integer expected_step required')
        fingerprint = seal({'action': action, 'expected_step': expected_step})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT digest,result FROM requests WHERE episode=? AND request_id=?', (eid, request_id)).fetchone()
            if prior:
                if prior[0] != fingerprint:
                    raise ValueError('Request ID reused with different content')
                return json.loads(prior[1])
            row = db.execute('SELECT config,actions,result FROM episodes WHERE id=?', (eid,)).fetchone()
            if not row:
                raise ValueError('Unknown episode')
            config, actions, last = map(json.loads, row)
            if last['result']['step'] != expected_step:
                raise ValueError('Stale step: resume the episode before issuing another decision')
            if last['result']['terminated'] or last['result']['truncated']:
                raise ValueError('Episode is finished')
            result = self.worker(config, actions+[action])
            if not result['ok']:
                db.execute('INSERT INTO interruptions VALUES (?,?,?)', (eid, time.time(), result['error']))
                return result
            result['episode_id'] = eid
            db.execute('UPDATE episodes SET actions=?,result=? WHERE id=?', (json.dumps(actions+[action]), json.dumps(result), eid))
            db.execute('INSERT INTO requests VALUES (?,?,?,?)', (eid, request_id, fingerprint, json.dumps(result)))
            return result


def serve(database, bank_path, host='127.0.0.1', port=8810, timeout=45):
    store = Store(database, json.loads(Path(bank_path).read_text()), timeout=timeout)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No episode capabilities, source, credentials or private URLs in logs.

        def do_POST(self):
            try:
                self.connection.settimeout(15)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if self.path == '/episodes':
                    result = store.create(body['task_id'], body.get('evidence', 'image'), body.get('mode', 'evaluation'))
                elif self.path == '/resume':
                    result = store.get(body['episode_id'])
                elif self.path == '/step':
                    result = store.step(body['episode_id'], body['request_id'], body['expected_step'], body['action'])
                else:
                    raise ValueError('Unknown endpoint')
                status = 200 if result.get('ok') else 503
            except (ValueError, KeyError, TypeError):
                status, result = 400, {'ok': False, 'error': 'invalid_request'}
            except Exception:
                status, result = 503, {'ok': False, 'error': 'server_interruption', 'scored': False, 'retryable': True}
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    ThreadingHTTPServer((host, port), Handler).serve_forever()
