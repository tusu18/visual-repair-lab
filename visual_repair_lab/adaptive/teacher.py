"""Persistent adaptive controller for an external learner's eight-attempt groups."""
import json,random
from .curriculum import Curriculum
from .generator import propose,edit_task,evaluate_proposal
from .pool_gate import certify_against_catalogue
from .dsl import seal


class AdaptiveTeacher:
    def __init__(self,database,policy_id,seed=7):
        self.curriculum=Curriculum(database,policy_id,seed=seed);self.policy_id=policy_id
        self.rng=random.Random(seed+997);self.serial=10000;self.rejections=[]
        with self.curriculum.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS teacher_state(policy TEXT PRIMARY KEY,state TEXT)')
            row=db.execute('SELECT state FROM teacher_state WHERE policy=?',(policy_id,)).fetchone()
            if row:
                state=json.loads(row[0]);r=state['rng'];self.rng.setstate((r[0],tuple(r[1]),r[2]))
                self.serial=state['serial'];self.rejections=state['rejections']

    def _save(self):
        with self.curriculum.connect() as db:
            db.execute('INSERT OR REPLACE INTO teacher_state VALUES (?,?)',(self.policy_id,json.dumps({
                'rng':self.rng.getstate(),'serial':self.serial,'rejections':self.rejections})))

    def _accept(self,spec,lineage):
        spec=dict(spec,id='teacher-'+seal(self.policy_id)[:10]+'-'+str(self.serial))
        record=evaluate_proposal(spec);catalogue=list(self.curriculum.snapshot().values())
        if record['certificate']['accepted']:
            c=certify_against_catalogue(record,catalogue+[record]);record['certificate']['whole_catalogue_gate']=c
            if not c['accepted']:record['certificate']['accepted']=False;record['certificate']['reasons']+=c['reasons']
        if record['certificate']['accepted']:
            signature=(record['certificate']['behavior_fingerprint'],spec['budget'],spec['starting_point'])
            existing={(v['certificate']['behavior_fingerprint'],v['spec']['budget'],v['spec']['starting_point']) for v in catalogue}
            if signature in existing:record['certificate']['accepted']=False;record['certificate']['reasons'].append('duplicate_effective_task')
        if record['certificate']['accepted']:self.curriculum.add(record,lineage)
        else:self.rejections.append({'spec':spec,'reasons':record['certificate']['reasons'],'lineage':lineage})
        self._save();return record

    def next_task(self,max_generation_attempts=25):
        selected,reason=self.curriculum.choose()
        if selected is not None:return selected,reason
        # All-zero pools are parked, never disguised as useful training groups.
        # First soften a parked task; then try genuinely new grammar proposals.
        for _ in range(max_generation_attempts):
            tasks=list(self.curriculum.snapshot().values())
            parked=[v for v in tasks if v.get('estimate',{}).get('p')==0]
            retired=[v for v in tasks if v.get('estimate',{}).get('state')=='retired']
            self.serial+=1
            parent=self.rng.choice(parked) if parked else self.rng.choice(retired) if retired else None
            child=edit_task(parent['spec'],parent['estimate']['p'],self.rng,self.serial) if parent else None
            if child:spec,lineage=child
            else:spec,lineage=propose(self.rng,self.serial),{'operation':'new_proposal'}
            record=self._accept(spec,lineage)
            if record['certificate']['accepted']:return record,{'reason':'generated_or_softened','lineage':lineage}
        return None,{'reason':'generation_exhausted_without_certified_task','attempts':max_generation_attempts}

    def observe_group(self,task_id,round_index,attempts):
        summary=self.curriculum.record(task_id,round_index,attempts)
        if summary.get('eligible') and (summary['p']==0 or summary['p']>=.75):
            parent=self.curriculum.snapshot()[task_id]['spec'];self.serial+=1
            child=edit_task(parent,summary['p'],self.rng,self.serial)
            if child:
                spec,lineage=child;record=self._accept(spec,lineage)
                summary['editor_child_accepted']=record['certificate']['accepted']
        self._save();return summary
