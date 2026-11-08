"""Frozen HTTP-only native GPT benchmark. Never regenerates completed decisions."""
import argparse,base64,concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'actor_client'))
from client import Client
from codex_transport import DiscoveryCodexPolicy

SCHEMA={'type':'object','additionalProperties':False,'properties':{'action_json':{'type':'string'},'summary':{'type':'string'}},'required':['action_json','summary']}


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def source_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('visual_repair_lab','vision_sort','actor_client')
            for p in sorted((root/sub).rglob('*.py'))}


def run_job(job,config,out):
    root=Path(config['source_root']);assert source_hashes(root)==config['source_hashes']
    folder=out/'trials'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'checkpoint.json';result_path=folder/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    client=Client(config['endpoint'],timeout=180);policy=DiscoveryCodexPolicy(model=job['model'],effort=config['effort'],timeout=240)
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());response=client.resume(state['episode_id'])
        assert state['job']==job
    else:
        response=client.create(job['task_id'],'image','evaluation')
        state={'job':job,'episode_id':response['episode_id'],'history':[],'pending':None}
        atomic(checkpoint,state)
    while True:
        if state['pending']:
            pending=state['pending'];response=client.step(state['episode_id'],pending['step'],pending['action'],pending['request_id'])
            if not response['ok']:
                r=dict(job,status='interrupted',reason='evaluator',steps=pending['step']);atomic(result_path,r);return r
            state['history'].append({'step':pending['step']+1,'action':pending['action'],'summary':pending['summary']})
            state['pending']=None;atomic(checkpoint,state)
        public=response['result'];step=public['step'];observation=folder/f'{step:03}-observation.json'
        if observation.exists():assert json.loads(observation.read_text())==public
        else:atomic(observation,public)
        image_path=folder/f'{step:03}.png';image_path.write_bytes(base64.b64decode(public['image_png_base64']))
        if public['terminated'] or public['truncated']:
            f=public['context'].get('final_result')
            r=dict(job,status='completed',success=bool(f and f['success']),final_result=f,
                   steps=step,truncated=public['truncated'])
            atomic(result_path,r);return r
        images=[image_path];contexts=[]
        if step:
            previous=json.loads((folder/f'{step-1:03}-observation.json').read_text())
            images=[folder/f'{step-1:03}.png',image_path]
            contexts=[{'image_index':0,'role':'preceding observation','step':step-1,
                       'observation_id':previous['context']['observation_id'],'source_revision':previous['context']['active_revision'],
                       'public_scene':previous['context']['scene']}]
        contexts.append({'image_index':len(images)-1,'role':'CURRENT observation','step':step,
                         'observation_id':public['context']['observation_id'],'source_revision':public['context']['active_revision']})
        prompt=('Repair the interactive software using only environment actions. Choose experiments, code edits and verification yourself. '
                'No external tools or local execution. Return one JSON action encoded in action_json and a short evidence-based summary. '
                'The LAST attached image is CURRENT; any earlier image is explicitly labelled below. '
                'Use the exact action examples in the public context. Do not infer hidden grader results from syntax success.\n'
                +'Public image bindings: '+json.dumps(contexts)+'\nCurrent public context: '+json.dumps(public['context'])
                +'\nPrevious actions and summaries: '+json.dumps(state['history']))
        call_path=folder/f'{step:03}-call.json'
        if call_path.exists():call=json.loads(call_path.read_text())
        else:
            assert source_hashes(root)==config['source_hashes']
            (folder/f'{step:03}-prompt.txt').write_text(prompt)
            call=policy.call(prompt,images,SCHEMA);atomic(call_path,call)
        if call['status']!='ok':
            r=dict(job,status='interrupted',reason=call['status'],steps=step);atomic(result_path,r);return r
        assert call['audit']['valid'] and call['audit']['tool_calls']==0
        try:
            answer=json.loads(call['completion_text'])
            try:action=json.loads(answer['action_json'])
            except ValueError:action=answer['action_json']  # Forward the actual invalid output; let the environment charge it.
        except (ValueError,KeyError,TypeError):
            r=dict(job,status='interrupted',reason='invalid_model_action_json',steps=step);atomic(result_path,r);return r
        state['pending']={'step':step,'action':action,'summary':answer['summary'],'request_id':client.new_request_id()};atomic(checkpoint,state)
        print(json.dumps({'job':job['id'],'step':step+1,'action':action.get('type') if isinstance(action,dict) else 'invalid'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lock=out/'SCHEDULER.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    except FileExistsError:raise SystemExit('A scheduler lock exists; do not run concurrently')
    try:
        jobs=[j for j in config['jobs'] if j['stage']==args.stage]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_job,j,config,out) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                r=future.result();print('FINISHED '+json.dumps(r),flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
