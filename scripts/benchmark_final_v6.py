"""Pre-registered final baseline. HTTP actor boundary, immutable decisions."""
import argparse,base64,concurrent.futures,getpass,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'actor_client'))
from client import Client
from codex_transport import DiscoveryCodexPolicy
from hosted_transport import HostedPolicy,SpendLedger
SCHEMA={'type':'object','additionalProperties':False,'properties':{'action_json':{'type':'string'},'summary':{'type':'string'}},'required':['action_json','summary']}

def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def hashes(root):
 return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ('visual_repair_lab','vision_sort','actor_client') for p in sorted((root/name).rglob('*.py'))}

def public_history(history):
 rows=[]
 for h in history:
  row=dict(h);action=h['action']
  if isinstance(action,dict) and action.get('type')=='edit' and isinstance(action.get('source'),str):
   row['action']={'type':'edit','source_sha256':hashlib.sha256(action['source'].encode()).hexdigest(),'source_bytes':len(action['source'].encode())}
  rows.append(row)
 return rows

def prompt_for(public,history,previous=None,active_source=None):
 context=dict(public['context']);source=context.pop('source','')
 keys=('environment','family','prompt','interactions','operations','language_contract','action_examples','reward_contract','level','image_access','evidence_arm')
 contract={k:context.pop(k) for k in keys if k in context}
 prefix=('Repair the interactive software using only environment actions. Choose experiments, code edits and verification yourself. '
  'No external tools or local execution. Return one JSON action encoded in action_json and a short evidence-based summary. '
  'The attached image is the CURRENT environment observation. Earlier images are not reattached; your public actions and summaries are retained. '
  'Use the exact action examples. A successful run checks syntax/schema, not reference correctness. '
  'Earlier edit bodies are hashes; current saved code and any different active code are supplied in full.\n'
  +'Public task contract: '+json.dumps(contract,separators=(',',':'))+'\nCurrent saved source:\n'+source)
 if active_source is not None:context['currently_active_source']=active_source
 bindings=[dict(image_index=0,role='CURRENT observation',step=public['step'],observation_id=context['observation_id'],active_revision=context['active_revision'])]
 return prefix+'\nCURRENT_PUBLIC_STATE_AND_HISTORY:\n'+json.dumps(context,separators=(',',':'))+'\nImage binding: '+json.dumps(bindings,separators=(',',':'))+'\nEarlier public actions and summaries: '+json.dumps(public_history(history),separators=(',',':'))

def run_job(job,profile,config,out,policy):
 folder=out/'trials'/job['id'];folder.mkdir(parents=True,exist_ok=True);result_file=folder/'result.json';checkpoint=folder/'checkpoint.json'
 if result_file.exists():return json.loads(result_file.read_text())
 client=Client(profile['endpoint'],timeout=240)
 if checkpoint.exists():state=json.loads(checkpoint.read_text());response=client.resume(state['episode_id'])
 else:
  response=client.create(job['task_id'],'image','evaluation')
  if not response.get('ok'):raise RuntimeError('evaluator_create')
  state={'job':job,'episode_id':response['episode_id'],'history':[],'pending':None};atomic(checkpoint,state)
 while True:
  if state['pending']:
   q=state['pending'];response=client.step(state['episode_id'],q['step'],q['action'],q['request_id'])
   if not response.get('ok'):raise RuntimeError('evaluator_step')
   state['history'].append({'step':q['step']+1,'action':q['action'],'summary':q['summary']});state['pending']=None;atomic(checkpoint,state)
  public=response['result'];n=public['step'];op=folder/f'{n:03}-observation.json'
  if op.exists():assert json.loads(op.read_text())==public
  else:atomic(op,public)
  ip=folder/f'{n:03}.png';ip.write_bytes(base64.b64decode(public['image_png_base64']))
  if public['terminated'] or public['truncated']:
   grade=public['context'].get('final_result');r=dict(job,status='completed',success=bool(grade and grade['success']),final_result=grade,steps=n,truncated=public['truncated'])
   atomic(result_file,r);return r
  active=None;context=public['context']
  if context['active_revision']!=context['source_revision']:
   for old in range(n,-1,-1):
    before=json.loads((folder/f'{old:03}-observation.json').read_text())['context']
    if before['source_revision']==context['active_revision']:
     active=before['source'];break
   assert active is not None
  images=[ip];prompt=prompt_for(public,state['history'],active_source=active);cp=folder/f'{n:03}-call.json'
  if cp.exists():call=json.loads(cp.read_text())
  else:
   assert hashes(Path(config['source_root']))==config['source_hashes'];(folder/f'{n:03}-prompt.txt').write_text(prompt)
   for attempt in range(3):
    call=policy.call(prompt,images,SCHEMA);atomic(folder/f'{n:03}-transport-{attempt:02}.json',call)
    if call['status'] in ('ok','budget_censored') or call.get('http_status') in (400,401,402,403,404):break
    if attempt<2:time.sleep(5)
   atomic(cp,call)
  if call['status']!='ok':
   r=dict(job,status='budget_censored' if call['status']=='budget_censored' else 'interrupted',reason=call['status'],steps=n);atomic(result_file,r);return r
  assert call['audit']['valid'] and call['audit']['tool_calls']==0
  # Model-format errors are model actions and cost an action; they are never
  # censored as infrastructure errors or repaired on the model's behalf.
  try:
   answer=json.loads(call['completion_text']);summary=answer.get('summary','')
   action=json.loads(answer['action_json'])
  except (ValueError,KeyError,TypeError,AttributeError):action=call['completion_text'];summary='Actual model output did not match the published action format.'
  state['pending']={'step':n,'action':action,'summary':summary,'request_id':client.new_request_id()};atomic(checkpoint,state)
  print(json.dumps({'job':job['id'],'step':n+1,'action':action.get('type') if isinstance(action,dict) else 'invalid_model_output'}),flush=True)

def run_profile(profile,config,out,key):
 model=profile['name'];ledger=SpendLedger(config['spending_database'])
 policy=HostedPolicy(key,profile,ledger) if profile['transport']=='openrouter' else DiscoveryCodexPolicy(model=profile['model'],effort='low',timeout=240)
 jobs=[j for j in config['jobs'] if j['model']==model]
 blocked=not profile.get('available',True)
 for job in jobs:
  if blocked:
   atomic(out/'trials'/job['id']/'result.json',dict(job,status='not_run_after_budget_or_access_block',scored=False));continue
  try:r=run_job(job,profile,config,out,policy)
  except Exception as exc:
   r=dict(job,status='interrupted',reason='runner_'+type(exc).__name__);atomic(out/'trials'/job['id']/'result.json',r)
  print('FINISHED '+json.dumps(r),flush=True)
  # Stop only this hosted profile if access or the authorized budget blocks it.
  # Native interruptions leave other worlds runnable; no provider fallback.
  if profile['transport']=='openrouter' and r['status']!='completed':blocked=True
  if profile['transport']=='openrouter':atomic(out/'spending-summary.json',ledger.summary())
 return model

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--read-key',action='store_true');p.add_argument('--profiles',nargs='*');a=p.parse_args()
 key=getpass.getpass('OpenRouter key (memory only): ') if a.read_key else os.environ.get('OPENROUTER_API_KEY','')
 config=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 assert hashes(Path(config['source_root']))==config['source_hashes']
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['runner_sha256']
 lock=out/'SCHEDULER.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(config['profiles'])) as pool:
   fs=[pool.submit(run_profile,q,config,out,key) for q in config['profiles'] if not a.profiles or q['name'] in a.profiles]
   for f in concurrent.futures.as_completed(fs):print('PROFILE_FINISHED',f.result(),flush=True)
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
