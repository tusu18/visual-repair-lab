"""Human-operated local VLM benchmark. No evaluator internals in model prompts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
from urllib.parse import urlparse
import uuid


def atomic(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def model_url(endpoint):
    parsed=urlparse(endpoint)
    if (parsed.scheme!='http' or parsed.hostname not in ('localhost','127.0.0.1','::1','host.docker.internal','host.containers.internal')
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('GUI models must use a local HTTP endpoint; remote paid calls are disabled')
    return endpoint.rstrip('/')+'/chat/completions'


def call_model(endpoint,model,key,public,history):
    content=[{'type':'text','text':
        'Repair the candidate to match the approved reference. Choose experiments and verification yourself. '
        'Return exactly one JSON object with action_json (a JSON-encoded action string) and summary (a short string). '
        'Use only published environment actions, no external tools.\nPUBLIC CONTEXT:\n'+
        json.dumps(public['context'])+'\nPAST ACTIONS:\n'+json.dumps(history)}]
    if public['context'].get('evidence_arm','image') in ('image','image_no_footer'):
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+public['image_png_base64']}})
    payload={'model':model,'messages':[{'role':'user','content':content}],
             'response_format':{'type':'json_object'},'max_tokens':4096,'stream':False}
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    wire=json.dumps(payload).encode()
    request=urllib.request.Request(model_url(endpoint),data=wire,headers=headers)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    with urllib.request.build_opener(NoRedirect).open(request,timeout=180) as response:answer=json.load(response)
    # Store public model response, never request headers or the supplied key.
    if key:answer=json.loads(json.dumps(answer).replace(key,'[REDACTED]'))
    return {'response':answer,'request_sha256':hashlib.sha256(wire).hexdigest(),
            'image_sha256':hashlib.sha256(public['image_png_base64'].encode()).hexdigest()}


def decode_call(call):
    choice=call['response']['choices'][0]
    if choice.get('finish_reason') not in ('stop',None) or choice['message'].get('tool_calls'):
        raise ValueError('Incomplete or tool-using model response; preserved without resampling')
    answer=json.loads(choice['message']['content'])
    action=json.loads(answer['action_json'])
    if type(action) is not dict or type(answer['summary']) is not str:
        raise ValueError('Invalid model action')
    return action,answer['summary']


class Benchmarks:
    def __init__(self,store,directory):
        self.store=store;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.guard=threading.Lock();self.running=set()

    def path(self,job):
        if len(job)!=32 or any(c not in '0123456789abcdef' for c in job):raise ValueError('Invalid run ID')
        return self.directory/job

    def start(self,body):
        model_url(body['endpoint'])
        if type(body['model']) is not str or not 1<=len(body['model'])<=200:raise ValueError('Model ID required')
        repetitions=body.get('repetitions',8);workers=body.get('workers',8)
        if type(repetitions) is not int or not 1<=repetitions<=8 or type(workers) is not int or not 1<=workers<=8:
            raise ValueError('Repetitions and concurrency must be 1–8')
        tasks=body.get('tasks',list(self.store.bank))
        if not isinstance(tasks,list) or not tasks or len(set(tasks))!=len(tasks) or any(t not in self.store.bank for t in tasks):
            raise ValueError('Select distinct available tasks')
        job=uuid.uuid4().hex;root=self.path(job);root.mkdir()
        config={'id':job,'model':body['model'],'endpoint':body['endpoint'],'tasks':tasks,
            'repetitions':repetitions,'workers':workers,'evidence':body.get('evidence','image'),
            'mode':'evaluation','runtime_bank_seal':self.store.seal,'created':time.time()}
        if config['evidence'] not in ('image','image_no_footer','numeric_text','none'):raise ValueError('Unknown evidence')
        atomic(root/'config.json',config)
        self.launch(job,body.get('key',''))
        return {'ok':True,'run_id':job}

    def launch(self,job,key=''):
        root=self.path(job);config=json.loads((root/'config.json').read_text())
        if config['runtime_bank_seal']!=self.store.seal:raise ValueError('Cannot resume under changed runtime or bank')
        with self.guard:
            if job in self.running:raise ValueError('Run already active')
            self.running.add(job)
        def run():
            try:
                plans=[(t,r) for t in config['tasks'] for r in range(config['repetitions'])]
                with ThreadPoolExecutor(max_workers=config['workers']) as pool:
                    list(pool.map(lambda plan:self.attempt(root,config,*plan,key),plans))
            finally:
                with self.guard:self.running.discard(job)
        threading.Thread(target=run,daemon=True).start()

    def attempt(self,root,config,task,repeat,key):
        index=config['tasks'].index(task)*config['repetitions']+repeat
        out=root/f'{index:04d}';out.mkdir(exist_ok=True);checkpoint=out/'checkpoint.json'
        state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'task':task,'repeat':repeat,'history':[],'pending':None}
        if state.get('status')=='completed':return
        try:
            if 'episode_id' not in state:
                response=self.store.create(task,config['evidence'])
                if not response['ok']:raise RuntimeError('Evaluator start interrupted')
                state['episode_id']=response['episode_id'];atomic(checkpoint,state)
            response=self.store.get(state['episode_id'])
            while True:
                if state['pending']:
                    p=state['pending'];response=self.store.step(state['episode_id'],p['id'],p['step'],p['action'])
                    if not response['ok']:raise RuntimeError('Evaluator step interrupted')
                    state['history'].append({'step':p['step']+1,'action':p['action'],'summary':p['summary']})
                    state['pending']=None;atomic(checkpoint,state)
                public=response['result'];step=public['step']
                atomic(out/f'{step:03d}-observation.json',public)
                if public['terminated'] or public['truncated']:
                    final=public['context'].get('final_result')
                    state.update(status='completed',success=bool(final and final['success']),
                        passed=final['passed'] if final else 0,total=final['total'] if final else None,
                        steps=step,total_return=public['total_return'])
                    atomic(checkpoint,state);return
                call_path=out/f'{step:03d}-call.json'
                # A completed malformed response is retained, never regenerated.
                if call_path.exists():call=json.loads(call_path.read_text())
                else:
                    call=call_model(config['endpoint'],config['model'],key,public,state['history']);atomic(call_path,call)
                action,summary=decode_call(call)
                state['pending']={'id':uuid.uuid4().hex,'step':step,'action':action,'summary':summary}
                atomic(checkpoint,state)
        except Exception as exc:
            state.update(status='interrupted',interruption=type(exc).__name__)
            atomic(checkpoint,state)

    def summary(self,job):
        root=self.path(job);config=json.loads((root/'config.json').read_text());rows=[]
        for p in sorted(root.glob('*/checkpoint.json')):
            r=json.loads(p.read_text());rows.append({k:v for k,v in r.items() if k not in ('pending','history','episode_id')})
        complete=[r for r in rows if r.get('status')=='completed']
        groups=[]
        for task in config['tasks']:
            attempts=[r for r in complete if r['task']==task]
            if len(attempts)==8:
                p=sum(r['success'] for r in attempts)/8
                groups.append({'task':task,'p':p,'mixed':0<p<1})
        return {'ok':True,'config':config,'running':job in self.running,'planned':len(config['tasks'])*config['repetitions'],
            'completed':len(complete),'full_repairs':sum(r['success'] for r in complete),
            'interrupted':sum(r.get('status')=='interrupted' for r in rows),
            'mean_partial':sum((r['passed']/r['total']) if r['total'] else 0 for r in complete)/len(complete) if complete else None,
            'eligible_groups':len(groups),'mixed_groups':sum(g['mixed'] for g in groups),
            'signal_share':sum(g['mixed'] for g in groups)/len(groups) if groups else None,'groups':groups,'attempts':rows,
            'note':'Repeated worlds are clustered; mixed outcomes are a diversity proxy, not proof of learning.'}
