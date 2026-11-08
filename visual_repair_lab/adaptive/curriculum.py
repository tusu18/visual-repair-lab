"""Eight-attempt task estimates, learnability replay and auditable filtering."""
import json
import hashlib
import random
import sqlite3
from pathlib import Path
from visual_repair_lab.v5.generator import wilson
from .dsl import seal

GROUP_SIZE=8


def environment_seal():
    root=Path(__file__).resolve().parents[2]
    excluded={'private_cv.py','private_checks.py','detection_reference.py','server.py','worker.py','factory.py'}
    return seal({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for name in ('visual_repair_lab','vision_sort') for p in sorted((root/name).rglob('*.py'))
                 if p.parent.name!='adaptive' or p.name not in excluded})


def summarize_group(attempts):
    if len({r.get('attempt_id') for r in attempts})!=len(attempts):
        raise ValueError('Attempt IDs must be present and unique')
    completed=[r for r in attempts if r.get('status')=='completed']
    if len(attempts)!=GROUP_SIZE or len(completed)!=GROUP_SIZE:
        return {'eligible':False,'reason':'incomplete_or_interrupted_group','completed':len(completed)}
    if any(type(r.get('success')) is not bool for r in completed):
        raise ValueError('Binary success outcome required')
    successes=sum(r['success'] for r in completed);p=successes/GROUP_SIZE
    rewards=[r.get('return',0.) for r in completed]
    return {'eligible':True,'successes':successes,'n':GROUP_SIZE,'p':p,'priority':p*(1-p),
        'mixed':0<successes<GROUP_SIZE,'binary_training_keep':0<successes<GROUP_SIZE,
        'state':'parked' if p==0 else 'retired' if p>.9 else 'active',
        'wilson_95':wilson(successes,GROUP_SIZE),'dense_rewards_vary':len(set(rewards))>1}


class Curriculum:
    def __init__(self,path,policy_id,seed=7,replay_probability=.05):
        self.path=str(path);self.policy_id=policy_id;self.rng=random.Random(seed);self.replay_probability=replay_probability
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,spec TEXT,certificate TEXT,lineage TEXT);
              CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY,policy TEXT,task TEXT,round INTEGER,attempts TEXT,summary TEXT);
              CREATE TABLE IF NOT EXISTS metadata(policy TEXT PRIMARY KEY,rng TEXT);
              CREATE TABLE IF NOT EXISTS environment_metadata(seal TEXT NOT NULL);''')
            current=environment_seal();old=db.execute('SELECT seal FROM environment_metadata').fetchone()
            if old and old[0]!=current:raise ValueError('Environment changed: preserve this curriculum and use a new database')
            if not old:
                if db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]:raise ValueError('Unsealed legacy curriculum: preserve it and use a new database')
                db.execute('INSERT INTO environment_metadata VALUES (?)',(current,))
            row=db.execute('SELECT rng FROM metadata WHERE policy=?',(policy_id,)).fetchone()
            if row:
                data=json.loads(row[0]);self.rng.setstate((data[0],tuple(data[1]),data[2]))

    def connect(self):
        db=sqlite3.connect(self.path);db.execute('PRAGMA synchronous=FULL');return db

    def add(self,record,lineage=None):
        spec=record['spec'];cert=record['certificate']
        if spec['split']!='train' or not cert['accepted'] or cert['spec_sha256']!=seal(spec):
            raise ValueError('Only certified train tasks can enter the curriculum')
        if not cert.get('whole_catalogue_gate',{}).get('accepted'):
            raise ValueError('A target-independent whole-catalogue certificate is required')
        with self.connect() as db:
            prior=db.execute('SELECT spec FROM tasks WHERE id=?',(spec['id'],)).fetchone()
            if prior and json.loads(prior[0])!=spec:raise ValueError('Task ID changed')
            db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,?)',(spec['id'],json.dumps(spec),json.dumps(cert),json.dumps(lineage)))

    def record(self,task_id,round_index,attempts):
        summary=summarize_group(attempts)
        gid=seal([self.policy_id,task_id,round_index,attempts])
        with self.connect() as db:
            if not db.execute('SELECT 1 FROM tasks WHERE id=?',(task_id,)).fetchone():raise ValueError('Unknown task')
            db.execute('INSERT OR IGNORE INTO groups VALUES (?,?,?,?,?,?)',(gid,self.policy_id,task_id,round_index,json.dumps(attempts),json.dumps(summary)))
        return summary

    def snapshot(self):
        with self.connect() as db:
            tasks={r[0]:{'spec':json.loads(r[1]),'certificate':json.loads(r[2])} for r in db.execute('SELECT id,spec,certificate FROM tasks')}
            for task_id,summary in db.execute('SELECT task,summary FROM groups WHERE policy=? ORDER BY rowid',(self.policy_id,)):
                parsed=json.loads(summary)
                if parsed.get('eligible'):tasks[task_id]['estimate']=parsed
        return tasks

    def choose(self):
        tasks=self.snapshot();unseen=[k for k,v in tasks.items() if 'estimate' not in v]
        active=[k for k,v in tasks.items() if v.get('estimate',{}).get('state')=='active']
        retired=[k for k,v in tasks.items() if v.get('estimate',{}).get('state')=='retired']
        if unseen and (not active or self.rng.random()<.2):
            choice=self.rng.choice(unseen);reason='unseen_calibration'
        elif retired and self.rng.random()<self.replay_probability:
            choice=self.rng.choice(retired);reason='occasional_retired_replay'
        elif active:
            choice=self.rng.choices(active,weights=[tasks[k]['estimate']['priority'] for k in active],k=1)[0];reason='p_times_one_minus_p'
        elif retired:
            return None,{'reason':'all_mastered_generate_or_harden','retired':len(retired)}
        else:return None,{'reason':'no_active_tasks_generate_or_edit','parked':len(tasks)}
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(self.policy_id,json.dumps(self.rng.getstate())))
        return tasks[choice],{'reason':reason,'task_id':choice}


def signal_summary(groups):
    eligible=[r for r in groups if r['summary'].get('eligible')]
    mixed=sum(r['summary']['mixed'] for r in eligible)
    kept=[r for r in eligible if r['summary']['binary_training_keep']]
    histogram={str(i/8):sum(r['summary']['successes']==i for r in eligible) for i in range(9)}
    return {'attempted_groups':len(groups),'eligible_groups':len(eligible),'mixed_groups':mixed,
        'raw_mixed_share':mixed/len(eligible) if eligible else None,'raw_mixed_wilson_95':wilson(mixed,len(eligible)),
        'kept_groups':len(kept),'kept_mixed_share':1. if kept else None,
        'dropped_groups':len(eligible)-len(kept),'pass_rate_histogram':histogram,
        'note':'Binary-outcome diversity proxy, not measured gradients or learning. Dense partial rewards can vary within all-fail groups.'}
