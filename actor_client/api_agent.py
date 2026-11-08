"""Live restricted GPT actor. Contains no task-world or reference access.

Run from an actor-only filesystem. Persist the pending decision before posting;
resume reuses that request ID, never regenerating a completed model decision.
"""
import argparse
import base64
import json
import os
from pathlib import Path
from client import Client
from codex_transport import DiscoveryCodexPolicy

SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'action_json':{'type':'string'},'summary':{'type':'string'}},'required':['action_json','summary']}


def atomic(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',default='http://127.0.0.1:8810')
    p.add_argument('--task-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='gpt-6.1-sol');p.add_argument('--effort',default='low')
    p.add_argument('--evidence',default='image');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);checkpoint=out/'checkpoint.json'
    client=Client(args.endpoint);policy=DiscoveryCodexPolicy(model=args.model,effort=args.effort,timeout=240)
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());eid=state['episode_id'];response=client.resume(eid)
    else:
        response=client.create(args.task_id,args.evidence);eid=response['episode_id']
        state={'episode_id':eid,'history':[],'pending':None,'model':args.model,'effort':args.effort,'evidence':args.evidence}
        atomic(checkpoint,state)
    if (state['model'],state['effort'],state['evidence'])!=(args.model,args.effort,args.evidence):
        raise SystemExit('Resume configuration differs from the original attempt')
    while True:
        if state['pending']:
            pending=state['pending']
            response=client.step(eid,pending['step'],pending['action'],pending['request_id'])
            if not response['ok']: raise SystemExit('Evaluator interruption; pending action preserved')
            state['history'].append({'step':pending['step']+1,'action':pending['action'],'summary':pending['summary']})
            state['pending']=None;atomic(checkpoint,state)
        public=response['result'];step=public['step']
        atomic(out/f'{step:03}-observation.json',public)
        if public['terminated'] or public['truncated']:
            atomic(out/'result.json',public);print(json.dumps({'step':step,'final_result':public['context'].get('final_result'),'truncated':public['truncated']}),flush=True);return
        image=out/f'{step:03}.png';image.write_bytes(base64.b64decode(public['image_png_base64']))
        prompt=('You are repairing interactive software in Visual Repair Lab. Use only the environment actions. '
                'Return action_json containing one valid JSON action and a short evidence-based summary. '
                'Choose experiments and verification yourself. No external code or tools. '
                'Public context: '+json.dumps(public['context'])+'\nPrevious actions and summaries: '+json.dumps(state['history']))
        (out/f'{step:03}-prompt.txt').write_text(prompt)
        call_path=out/f'{step:03}-call.json'
        if call_path.exists():
            # Crash after model response but before pending-checkpoint write:
            # recover the saved decision rather than generating it a second time.
            call=json.loads(call_path.read_text())
        else:
            call=policy.call(prompt,[image] if args.evidence in ('image','image_no_footer') else [],SCHEMA)
            atomic(call_path,call)
        if call['status']!='ok': raise SystemExit('Model interrupted; prefix preserved, no action invented')
        answer=json.loads(call['completion_text']);action=json.loads(answer['action_json'])
        state['pending']={'step':step,'action':action,'summary':answer['summary'],'request_id':client.new_request_id()}
        atomic(checkpoint,state)
        print(json.dumps({'step':step+1,'action':action.get('type')}),flush=True)


if __name__=='__main__':main()
