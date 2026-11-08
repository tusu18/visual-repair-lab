"""Evaluator-only held-out family. Never imported by the train generator."""
import copy
import inspect
import json
import io
import base64
import hashlib
import string
import random
import gymnasium as gym
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from vision_sort.visual_training.runtime import checked_run,language_contract
from visual_repair_lab.v5.compare import equivalent
from .dsl import seal
from .detection_reference import detection_reference


def overlay(state,xywh,center,confidence,nms,swap,clip):
    frame=state['frame'];width,height=frame['width'],frame['height']
    scale=640/max(width,height)
    px=(640-width*scale)/2 if center else 0
    py=(640-height*scale)/2 if center else 0
    rows=[]
    ordered=state['detections'][:]
    for i in range(len(ordered)):
        for j in range(i+1,len(ordered)):
            if ordered[j]['score']>ordered[i]['score']:
                ordered[i],ordered[j]=ordered[j],ordered[i]
    for item in ordered:
        if not item['enabled'] or item['score']<confidence:
            continue
        a,b,c,d=item['box']
        if xywh:c,d=a+c,b+d
        box=[(a-px)/scale,(b-py)/scale,(c-px)/scale,(d-py)/scale]
        if clip:box=[max(0,min(width,box[0])),max(0,min(height,box[1])),max(0,min(width,box[2])),max(0,min(height,box[3]))]
        suppress=False
        for previous in rows:
            if previous['class_id']!=item['class_id']:continue
            x0,y0,x1,y1=previous['box']
            ix=max(0,min(x1,box[2])-max(x0,box[0]));iy=max(0,min(y1,box[3])-max(y0,box[1]))
            inter=ix*iy
            union=max(0,x1-x0)*max(0,y1-y0)+max(0,box[2]-box[0])*max(0,box[3]-box[1])-inter
            if union>0 and inter/union>nms:suppress=True
        if not suppress:
            rows.append({'id':item['id'],'class_id':item['class_id'],'box':box,
                         'label':('defect' if item['class_id']==0 else 'part') if swap else ('part' if item['class_id']==0 else 'defect')})
    out=[]
    for key in sorted([r['id'] for r in rows]):
        for r in rows:
            if r['id']==key:out.append({'id':r['id'],'box':r['box'],'label':r['label']})
    return {'objects':out}


def gold_source(spec):
    p=spec['pipeline']
    return inspect.getsource(overlay)+'\n\ndef solve(state):\n    return overlay(state, '+', '.join(repr(p[k]) for k in ('xywh','center','confidence','nms','swap','clip'))+')\n'


def initial_state(scene_seed=None):
    state={'family':'detection_overlay','frame':{'width':640,'height':360},'events':[],
            'detections':[{'id':'D1','box':[80.,180.,180.,100.],'score':.9,'class_id':0,'enabled':True},
                          {'id':'D2','box':[105.,195.,190.,110.],'score':.8,'class_id':0,'enabled':True},
                          {'id':'D3','box':[500.,80.,180.,160.],'score':.55,'class_id':1,'enabled':True},
                          {'id':'D4','box':[30.,560.,180.,120.],'score':.35,'class_id':1,'enabled':True}]}
    # Preserve all earlier evaluated worlds. New benchmark scenes vary actual
    # public geometry/scores, rather than counting identical scenes as seeds.
    if scene_seed is not None and scene_seed>=3514000:
        rng=random.Random(scene_seed);state['scene_seed']=scene_seed
        width,height=rng.choice(((600,360),(640,400),(720,440),(560,380)))
        state['frame']={'width':width,'height':height}
        x,y=rng.uniform(55,105),rng.uniform(155,205)
        w,h=rng.uniform(165,205),rng.uniform(85,125)
        boxes=[[x,y,w,h],[x+rng.uniform(18,35),y+rng.uniform(6,18),w+rng.uniform(0,20),h+rng.uniform(0,18)],
               [rng.uniform(485,530),rng.uniform(60,95),rng.uniform(160,200),rng.uniform(135,175)],
               [rng.uniform(15,50),rng.uniform(540,580),rng.uniform(165,195),rng.uniform(105,135)]]
        for row,box in zip(state['detections'],boxes):
            row['box']=[round(v,3) for v in box];row['score']=round(row['score']+rng.uniform(-.015,.015),4)
    return state


CONTROLS={'set_frame':{'width':[280,800],'height':[280,800]},'shift_scores':{'delta':[-.3,.3]},
          'set_overlap':{'offset':[-100,200]},'toggle_detection':{'id':['D1','D2','D3','D4'],'enabled':[False,True]},'reset_scene':{}}


def transition(state,event):
    kind=event.get('kind');bounds=CONTROLS.get(kind)
    if bounds is None or set(event)!={'kind',*bounds}:raise ValueError('Invalid event')
    for k,choices in bounds.items():
        if type(choices[0]) in (str,bool):
            if type(event[k]) is not type(choices[0]) or event[k] not in choices:raise ValueError('Invalid option')
        elif type(event[k]) not in (int,float) or not choices[0]<=event[k]<=choices[1]:raise ValueError('Invalid number')
    s=copy.deepcopy(state)
    if kind=='reset_scene':s=initial_state(state.get('scene_seed'))
    elif kind=='set_frame':s['frame']={k:event[k] for k in ('width','height')}
    elif kind=='shift_scores':
        for d in s['detections']:d['score']=min(1,max(0,d['score']+event['delta']))
    elif kind=='set_overlap':
        anchor=s['detections'][0]['box']
        s['detections'][1]['box'][0]=anchor[0]+event['offset']
        s['detections'][1]['box'][1]=anchor[1]+abs(event['offset'])*.05
    else:
        for d in s['detections']:
            if d['id']==event['id']:d['enabled']=event['enabled']
    s['events'].append(copy.deepcopy(event));return s


def states(scene_seed=None,grading_seed=None):
    result=[initial_state(scene_seed)]
    for w,h in ((640,360),(360,640),(480,480),(800,400)):
        for delta in (-.2,0.,.2):
            for offset in (0.,60.,180.):
                s=transition(initial_state(scene_seed),dict(kind='set_frame',width=w,height=h))
                s=transition(s,dict(kind='shift_scores',delta=delta))
                s=transition(s,dict(kind='set_overlap',offset=offset));result.append(s)
    if grading_seed is not None:
        rng=random.Random(grading_seed)
        for _ in range(24):
            s=transition(initial_state(scene_seed),dict(kind='set_frame',width=round(rng.uniform(280,800),3),height=round(rng.uniform(280,800),3)))
            delta=rng.uniform(-.3,.3)
            if rng.random()<.5:
                row=rng.choice(s['detections']);delta=max(-.3,min(.3,rng.choice((.45,.65))-row['score']+rng.choice((-1e-5,1e-5))))
            s=transition(s,dict(kind='shift_scores',delta=delta))
            s=transition(s,dict(kind='set_overlap',offset=round(rng.uniform(-100,200),3)))
            if rng.random()<.5:s=transition(s,dict(kind='toggle_detection',id=rng.choice(('D1','D2','D3','D4')),enabled=False))
            result.append(s)
    return result


def valid(value):
    if type(value) is not dict or set(value)!={'objects'} or type(value['objects']) is not list or len(value['objects'])>4:return False
    try:
        json.dumps(value,allow_nan=False)
        return len({r['id'] for r in value['objects']})==len(value['objects']) and all(type(r) is dict and set(r)=={'id','box','label'} and r['id'] in ('D1','D2','D3','D4') and type(r['label']) is str and
                   type(r['box']) is list and len(r['box'])==4 and all(type(x) in (int,float) for x in r['box']) for r in value['objects'])
    except (KeyError,TypeError,ValueError):return False


def objects_equivalent(a,b):
    """Detection collections have identity, but no public list-order contract."""
    if not valid(a) or not valid(b):return False
    return equivalent({'objects':sorted(a['objects'],key=lambda r:r['id'])},
                      {'objects':sorted(b['objects'],key=lambda r:r['id'])})


def panel(state,value,heading):
    im=Image.new('RGB',(600,700),'#f3f6fc');d=ImageDraw.Draw(im);font=ImageFont.load_default(size=16)
    d.text((20,18),heading,fill='#142b48',font=ImageFont.load_default(size=22))
    d.text((20,54),'Detection overlay repair',fill='#526783',font=font)
    w,h=state['frame']['width'],state['frame']['height'];scale=min(540/w,420/h)
    d.rectangle((30,100,30+w*scale,100+h*scale),fill='#dfe5ed',outline='#9baec3')
    # Procedural camera-like substrate, identical in both panels; no label boxes.
    for x,y in ((.12,.25),(.45,.48),(.72,.24)):
        d.rectangle((30+x*w*scale,100+y*h*scale,65+x*w*scale,125+y*h*scale),fill='#b6bdc9')
    if valid(value):
        for r in value['objects']:
            x0,y0,x1,y1=r['box'];color='#df6c28' if r['label']=='defect' else '#247d58'
            if x1>=x0 and y1>=y0:
                d.rectangle((30+x0*scale,100+y0*scale,30+x1*scale,100+y1*scale),outline=color,width=3)
            d.text((35+x0*scale,103+y0*scale),r['id']+' '+r['label'],fill=color,font=font)
        for i,r in enumerate(value['objects']):
            d.text((12,580+i*25),r['id']+' '+r['label']+' '+str([round(v,1) for v in r['box']]),fill='#142b48',font=font)
    else:d.text((20,580),'Invalid candidate output',fill='#b33737',font=font)
    return np.asarray(im)


STARTER='''def solve(state):
    out = []
    for d in state['detections']:
        if d['enabled'] and d['score'] >= 0.5:
            out.append({'id': d['id'], 'box': d['box'][:], 'label': 'class' + str(d['class_id'])})
    return {'objects': out}
'''


class DetectionRepairEnv(gym.Env):
    def __init__(self,spec,reward_mode='evaluation',evidence='image'):
        if spec['family']!='detection_overlay' or spec['split']!='private_test':raise ValueError('Private authority only')
        if reward_mode!='evaluation':raise ValueError('No training on the private family')
        self.spec=copy.deepcopy(spec);self.max_steps=spec['budget'];self.evidence=evidence
        self.family='detection_overlay';self.mode=reward_mode
        self.gold_source=gold_source(spec)
        self.solve_gold=lambda state:overlay(state,**self.spec['pipeline'])
        self.action_space=gym.spaces.Text(max_length=20000,charset=string.printable)
        self.observation_space=gym.spaces.Dict({'context':gym.spaces.Text(max_length=200000,charset=string.printable),
            'image':gym.spaces.Box(0,255,(700,1200,3),dtype=np.uint8)})

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.state=initial_state(self.spec['scene_seed']);self.source=self.active_source=self.starter=STARTER
        self.source_revision=self.active_revision=self.steps=0;self.terminated=self.truncated=False
        self.final_result=None;self.task_return=self.process_return=self.reward=0.;self.reward_ledger=[];self.action_ledger=[]
        self.verification=[];self._execute();return self._observation(),{}

    def _execute(self):
        r=checked_run(self.active_source,self.state);self.output=r['result'] if r['ok'] and valid(r['result']) else None
        self.public_result={'executed':r['ok'],'schema_valid':self.output is not None,'execution_status':r['error_code'],'error':'' if r['ok'] else r['error']}

    def _context(self):
        result=None if self.final_result is None else {k:self.final_result[k] for k in ('success','passed','total','final_revision_activated','chosen_verification_tests')}
        c={'environment':'VisualAdaptiveRepair-v0','family':self.family,'prompt':'Repair the detector overlay postprocessing to match the approved reference across available interactions. Raw detector rows are public. Use solve(state). No required experiment sequence. Every action costs budget; edit saves, run activates, submit grades saved code and ends the episode.',
           'scene':self.state,'source':self.source,'source_revision':self.source_revision,'active_revision':self.active_revision,
           'steps':self.steps,'budget':self.max_steps,'remaining':self.max_steps-self.steps,'public_result':self.public_result,
           'terminated':self.terminated,'truncated':self.truncated,'final_result':result,'interactions':CONTROLS,
           'operations':['observe','interact','edit','run','submit'],'observation_id':seal([self.state,self.steps,self.active_source])[:24],
           'language_contract':language_contract(),'action_examples':{'interact':{'type':'interact','event':{'kind':'set_frame','width':360,'height':640}},
           'edit':{'type':'edit','source':STARTER},'run':{'type':'run'},'submit':{'type':'submit'}}}
        return c

    def _observation(self):
        ref=self.solve_gold(copy.deepcopy(self.state));im=np.concatenate([panel(self.state,ref,'Approved reference'),panel(self.state,self.output,'Current candidate')],axis=1)
        c=self._context()
        if self.evidence=='image_no_footer':im[550:,:,:]=np.array([243,246,252],dtype=np.uint8)
        if self.evidence=='numeric_text':c['displayed_values']={'reference':ref,'candidate':self.output}
        if self.evidence in ('none','numeric_text'):im=np.zeros_like(im)
        return {'context':json.dumps(c),'image':im}

    def _grade(self,source):
        from .certify import batch
        cases=states(self.spec['scene_seed'],self.spec.get('grading_seed'));runs=batch(source,cases);checks=[]
        for i,(s,r) in enumerate(zip(cases,runs)):
            checks.append({'case':i,'passed':r['ok'] and objects_equivalent(r['result'],detection_reference(s,self.spec['pipeline']))})
        active=self.source_revision==self.active_revision
        return {'success':all(r['passed'] for r in checks),'passed':sum(r['passed'] for r in checks),'total':len(checks),'checks':checks,
                'final_revision_activated':active,'chosen_verification_tests':len(self.verification) if active else 0}

    def step(self,action):
        if self.terminated or self.truncated:raise gym.error.ResetNeeded('Episode completed')
        self.steps+=1;c={'action':-.025,'invalid':0.,'terminal':0.,'unactivated_submission':0.,'budget_exhaustion':0.}
        try:
            kind=action['type']
            if kind=='interact':
                self.state=transition(self.state,action['event']);self._execute()
                if self.source_revision and self.active_revision==self.source_revision and self.public_result['schema_valid']:
                    self.verification.append({'step':self.steps,'matches':objects_equivalent(self.output,self.solve_gold(copy.deepcopy(self.state)))})
            elif kind=='edit':
                source=action['source']
                if type(source) is not str or len(source.encode())>16000:raise ValueError('Invalid source')
                if source!=self.source:self.source=source;self.source_revision+=1;self.verification=[]
            elif kind=='run':self.active_source=self.source;self.active_revision=self.source_revision;self._execute()
            elif kind=='submit':
                self.final_result=self._grade(self.source);self.terminated=True
                f=self.final_result;c['terminal']=10. if f['success'] else -2+8*f['passed']/f['total']
                c['unactivated_submission']=-.25 if self.active_revision!=self.source_revision else 0.
            elif kind!='observe':raise ValueError('Unknown operation')
        except (KeyError,ValueError,TypeError):c['invalid']=-.1;kind='invalid'
        self.truncated=self.steps>=self.max_steps and not self.terminated
        if self.truncated:c['budget_exhaustion']=-2.
        self.reward=sum(c.values());self.task_return+=self.reward;self.action_ledger.append(copy.deepcopy(action))
        self.reward_ledger.append({'step':self.steps,'operation':kind,'components':c,'reward':self.reward,'cumulative_reward':self.task_return})
        return self._observation(),self.reward,self.terminated,self.truncated,{}

    def research_report(self):
        return {'version':'6.0.0','family':self.family,'actions':self.action_ledger,'reward_ledger':self.reward_ledger,
                'final_result':self.final_result,'steps':self.steps,'total_return':self.task_return,'verification':self.verification,
                'final_source_sha256':hashlib.sha256(self.source.encode()).hexdigest()}
