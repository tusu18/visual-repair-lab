"""Durable incremental workers with short optimistic SQLite commits."""
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from visual_repair_lab.adaptive.dsl import seal
from visual_repair_lab.adaptive.server import runtime_seal
from visual_repair_lab.adaptive.dsl import validate
from visual_repair_lab.adaptive.private_checks import validate_private


class Conflict(ValueError):
    pass


class Store:
    def __init__(self,path,bank,timeout=45):
        self.path,self.timeout=str(path),timeout
        Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        self.bank={r['spec']['id']:r['spec'] for r in bank['records'] if r['certificate']['accepted']}
        if not self.bank or seal(bank['records'])!=bank['bank_sha256']:
            raise ValueError('Invalid bank seal')
        for r in bank['records']:
            (validate_private if r['spec']['family']=='detection_overlay' else validate)(r['spec'])
            if r['certificate']['spec_sha256']!=seal(r['spec']):
                raise ValueError('Invalid certificate binding')
        self.runtime=runtime_seal()
        self.seal=seal({'runtime':self.runtime,'bank':bank['bank_sha256'],'schema':'v7-checkpoints-1'})
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''CREATE TABLE IF NOT EXISTS metadata(seal TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,config TEXT NOT NULL,
                actions TEXT NOT NULL,result TEXT NOT NULL,checkpoint TEXT NOT NULL,step INTEGER NOT NULL);
              CREATE TABLE IF NOT EXISTS requests(episode TEXT,request_id TEXT,digest TEXT,result TEXT,
                PRIMARY KEY(episode,request_id));
              CREATE TABLE IF NOT EXISTS interruptions(episode TEXT,at REAL,kind TEXT);''')
            prior=db.execute('SELECT seal FROM metadata').fetchone()
            if prior and prior[0]!=self.seal:
                raise ValueError('Runtime/bank changed: use a new database; old evidence is preserved')
            if not prior: db.execute('INSERT INTO metadata VALUES (?)',(self.seal,))
        os.chmod(self.path,0o600)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()

    def worker(self,config,snapshot=None,action=None):
        if runtime_seal()!=self.runtime:
            return {'ok':False,'error':'runtime_changed','scored':False,'retryable':False}
        env={k:v for k,v in os.environ.items() if k not in ('OPENROUTER_API_KEY','OPENAI_API_KEY')}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            p=subprocess.run([sys.executable,'-m','visual_repair_lab.v7.worker'],
                input=json.dumps({'config':config,'snapshot':snapshot,'action':action}),
                capture_output=True,text=True,timeout=self.timeout,env=env)
            value=json.loads(p.stdout)
            if not p.returncode and value.get('ok') and 'checkpoint' in value:return value
        except subprocess.TimeoutExpired:
            return {'ok':False,'error':'evaluation_timeout','scored':False,'retryable':True}
        except (OSError,ValueError):
            pass
        return {'ok':False,'error':'evaluator_interruption','scored':False,'retryable':True}

    def create(self,task_id,evidence='image',mode='evaluation'):
        if task_id not in self.bank or evidence not in ('image','numeric_text','image_no_footer','none') or mode not in ('train','evaluation'):
            raise ValueError('Unknown task or condition')
        if mode=='train' and self.bank[task_id]['split']!='train':
            raise ValueError('Training feedback restricted to train split')
        config={'spec':self.bank[task_id],'evidence':evidence,'mode':mode}
        evaluated=self.worker(config)
        if not evaluated['ok']:return evaluated
        snapshot=evaluated.pop('checkpoint');eid=secrets.token_urlsafe(32)
        public=dict(evaluated,episode_id=eid)
        with self.connect() as db:
            db.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)',
                (eid,json.dumps(config),'[]',json.dumps(public),json.dumps(snapshot),0))
        return public

    def get(self,eid):
        with self.connect() as db:row=db.execute('SELECT result FROM episodes WHERE id=?',(eid,)).fetchone()
        if not row:raise ValueError('Unknown episode')
        return json.loads(row[0])

    def _prior(self,db,eid,request_id,fingerprint):
        row=db.execute('SELECT digest,result FROM requests WHERE episode=? AND request_id=?',(eid,request_id)).fetchone()
        if row:
            if row[0]!=fingerprint:raise Conflict('Request ID reused with different content')
            return json.loads(row[1])

    def step(self,eid,request_id,expected_step,action):
        if not isinstance(request_id,str) or not 1<=len(request_id)<=100 or type(expected_step) is not int:
            raise ValueError('request_id and integer expected_step required')
        fingerprint=seal({'action':action,'expected_step':expected_step})
        # Read a consistent state, then close the connection BEFORE executing code.
        with self.connect() as db:
            prior=self._prior(db,eid,request_id,fingerprint)
            if prior:return prior
            row=db.execute('SELECT config,checkpoint,result,step FROM episodes WHERE id=?',(eid,)).fetchone()
        if not row:raise ValueError('Unknown episode')
        config,snapshot,last=map(json.loads,row[:3])
        if row[3]!=expected_step:raise Conflict('Stale step: resume before making another decision')
        if last['result']['terminated'] or last['result']['truncated']:raise Conflict('Episode finished')
        evaluated=self.worker(config,snapshot,action)
        if not evaluated['ok']:
            with self.connect() as db:
                db.execute('INSERT INTO interruptions VALUES (?,?,?)',(eid,time.time(),evaluated['error']))
            return evaluated
        snapshot=evaluated.pop('checkpoint');public=dict(evaluated,episode_id=eid)
        if public['result']['step']!=expected_step+1:raise ValueError('Invalid worker transition')
        # Compare-and-swap: a race can compute twice, but only one action commits.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=self._prior(db,eid,request_id,fingerprint)
            if prior:return prior
            row=db.execute('SELECT step,actions FROM episodes WHERE id=?',(eid,)).fetchone()
            if not row or row[0]!=expected_step:raise Conflict('Concurrent step won; resume the episode')
            actions=json.loads(row[1])+[action]
            db.execute('UPDATE episodes SET step=?,actions=?,result=?,checkpoint=? WHERE id=? AND step=?',
                (expected_step+1,json.dumps(actions),json.dumps(public),json.dumps(snapshot),eid,expected_step))
            db.execute('INSERT INTO requests VALUES (?,?,?,?)',(eid,request_id,fingerprint,json.dumps(public)))
        return public
