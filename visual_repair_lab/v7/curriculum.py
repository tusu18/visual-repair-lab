"""Optional aged scheduling priors, separate from measured checkpoint outcomes."""
import json
from visual_repair_lab.adaptive.curriculum import Curriculum as ColdCurriculum
from visual_repair_lab.adaptive.dsl import seal


class Curriculum(ColdCurriculum):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS warm_priors(policy TEXT,task TEXT,data TEXT,PRIMARY KEY(policy,task))')

    def inherit(self,source_policy,age=1,decay=.5):
        if source_policy==self.policy_id or type(age) is not int or age<1 or not 0<decay<1:
            raise ValueError('A different policy, positive age and decay in (0,1) required')
        latest={}
        with self.connect() as db:
            for task,gid,summary in db.execute('SELECT task,id,summary FROM groups WHERE policy=? ORDER BY rowid',(source_policy,)):
                row=json.loads(summary)
                if row.get('eligible'):latest[task]=(gid,row)
            for task,(gid,row) in latest.items():
                weight=decay**age;alpha=1+weight*row['successes'];beta=1+weight*(row['n']-row['successes'])
                prior={'source_policy':source_policy,'source_group':gid,'age':age,'decay':decay,
                    'effective_attempts':row['n']*weight,'p_prior':alpha/(alpha+beta),
                    'actual_current_attempts':0,'scheduling_only':True}
                db.execute('INSERT OR REPLACE INTO warm_priors VALUES (?,?,?)',(self.policy_id,task,json.dumps(prior)))
        return {'inherited':len(latest),'policy':self.policy_id,'source_policy':source_policy,'scheduling_only':True}

    def snapshot(self):
        tasks=super().snapshot()
        with self.connect() as db:
            for task,data in db.execute('SELECT task,data FROM warm_priors WHERE policy=?',(self.policy_id,)):
                if task in tasks:tasks[task]['prior']=json.loads(data)
        return tasks

    def choose(self):
        tasks=self.snapshot()
        warm=[k for k,v in tasks.items() if 'estimate' not in v and 'prior' in v]
        if warm and self.rng.random()<.8:
            choice=self.rng.choices(warm,weights=[tasks[k]['prior']['p_prior']*(1-tasks[k]['prior']['p_prior']) for k in warm],k=1)[0]
            with self.connect() as db:
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(self.policy_id,json.dumps(self.rng.getstate())))
            return tasks[choice],{'reason':'aged_prior_needs_current_measurement','task_id':choice,
                'prior_sha256':seal(tasks[choice]['prior']),'current_attempts':0}
        return super().choose()
