"""Standalone research figures and human-only public-observation replay."""
import argparse,json,collections,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.evidence).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 study=json.loads((root/'proxy-final/summary.json').read_text());generation=json.loads((root/'generation/pool.json').read_text());pool=json.loads((root/'generation/whole-catalogue-pool.json').read_text())
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
 colors={'fixed':'#6b7280','adaptive':'#2166ac'}
 for arm,label in (('fixed','Fixed pool'),('adaptive','Adaptive')):
  rows=[r for r in study['rounds'] if r['arm']==arm]
  axes[0].plot([r['round'] for r in rows],[100*r['raw_mixed_share'] for r in rows],marker='o',color=colors[arm],label=label)
  bins=study['arms'][arm]['pass_rate_histogram'];xs=[float(k) for k in bins]
  shift=-.025 if arm=='fixed' else .025
  axes[1].bar([x+shift for x in xs],list(bins.values()),width=.047,color=colors[arm],label=label)
 axes[0].set(xlabel='Prospective round (12 groups per arm)',ylabel='Mixed-outcome groups (%)',ylim=(0,100),xticks=[1,2,3,4])
 axes[1].set(xlabel='Group pass rate (8 attempts)',ylabel='Number of groups',xticks=[0,.25,.5,.75,1])
 axes[0].legend(frameon=False);axes[1].legend(frameon=False);fig.suptitle('Scripted train-pool diagnostic — no weight updates',fontsize=13)
 fig.savefig(out/'curriculum.png',dpi=180);fig.savefig(out/'curriculum.pdf');plt.close(fig)
 counts=collections.Counter(generation['rejection_counts'])
 for r in pool['rejected']:
  for reason in r['certificate']['whole_catalogue_gate']['reasons']:counts[reason]+=1
 labels={'rule_effect:0':'Rule without enough effect','rule_effect:1':'Rule without enough effect','rule_effect:2':'Rule without enough effect',
         'rounded_or_probe_alias':'Observation ambiguity','ideal_budget_two_spare':'Local ideal budget','whole_catalogue_budget':'Whole-catalogue ideal budget','duplicate_behavior':'Duplicate behavior'}
 grouped=collections.Counter()
 for k,v in counts.items():grouped[labels.get(k,k)]+=v
 fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');ys=list(grouped)
 ax.barh(ys,[grouped[k] for k in ys],color='#6b7280');ax.set(xlabel='Rejected proposals',title=f'200 proposed → {len(pool["records"])} accepted')
 for i,k in enumerate(ys):ax.text(grouped[k]+.4,i,str(grouped[k]),va='center')
 fig.savefig(out/'rejections.png',dpi=180);plt.close(fig)
 # Latest transport-continuation records are additive to the preserved original.
 runs={}
 for folder in (root/'gpt-study',root/'gpt-private-continuation',root/'gpt-private-continuation-2',root/'gpt-native-adaptive'):
  if not (folder/'trials').exists():continue
  for rp in (folder/'trials').glob('*/result.json'):
   result=json.loads(rp.read_text());key=result['id']
   if key not in runs or (result['status']=='completed' and runs[key]['result']['status']!='completed'):
    runs[key]={'result':result,'folder':rp.parent}
 replay=[]
 for key,item in sorted(runs.items()):
  folder=item['folder'];steps=[]
  for op in sorted(folder.glob('*-observation.json')):
   o=json.loads(op.read_text());n=o['step'];context=o['context'];checkpoint=json.loads((folder/'checkpoint.json').read_text())
   history=checkpoint['history'];entry=next((h for h in history if h['step']==n),None)
   steps.append({'step':n,'image':str((folder/f'{n:03}.png').relative_to(root)),'reward':o['reward'],
                 'action':entry['action'] if entry else None,'summary':entry['summary'] if entry else 'Initial observation',
                 'source':context.get('source',''),'remaining':context['remaining']})
  replay.append({'id':key,'result':item['result'],'steps':steps})
 data={'study':study,'accepted':len(pool['records']),'proposed':200,'rejections':dict(grouped),'runs':replay}
 (out/'data.json').write_text(json.dumps(data,indent=2))
 payload=json.dumps(data).replace('<','\\u003c')
 html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual Repair Lab · Adaptive study</title>
 <style>body{font:16px system-ui;margin:0;color:#17263a;background:#f6f8fb}main{max-width:1120px;margin:auto;padding:32px}h1{font-size:30px;margin-bottom:6px}h2{margin-top:38px;font-size:22px}.muted{color:#526277}.metrics{display:flex;gap:32px;flex-wrap:wrap;padding:24px 0}.metric b{font-size:30px;display:block}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #d9e0e9;padding:10px}select,button,input{font:inherit;padding:8px}input[type=range]{width:75%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e9eef5;padding:16px;border-radius:8px;font:13px ui-monospace,monospace}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.note{max-width:900px;line-height:1.6}</style>
 <main><h1>Visual Repair Lab · Adaptive generator</h1><div class="muted">V6 research delivery · scripted curriculum diagnostics and private GPT evaluation</div>
 <div class="metrics"><div class="metric"><b id="accepted"></b>Accepted generated tasks</div><div class="metric"><b id="adaptive"></b>Adaptive mixed groups</div><div class="metric"><b id="fixed"></b>Fixed mixed groups</div></div>
 <p class="note">Mixed groups contain both full repairs and failures across eight attempts. These headline figures are measured before filtering, using a frozen scripted catalogue-search agent with numeric observations. No model weights were updated. The histogram still includes many easy groups; a stable 50% pass frontier has not been established.</p>
 <img src="curriculum.png" alt="Mixed group share over rounds and pass-rate histogram"><h2>Generator rejection report</h2><img src="rejections.png" alt="Generated task rejection counts">
 <h2>Private model evaluation</h2><p class="note">Only the held-out detection-overlay worlds appear in this score table. Train calibration episodes are sampling diagnostics. Interruptions are excluded. Repeated attempts on two worlds do not establish a general model ranking.</p><table><thead><tr><th>Model</th><th>Task</th><th>Full repairs</th><th>Mean checks passed</th><th>Interrupted</th></tr></thead><tbody id="scores"></tbody></table>
 <h2>Actual agent replay</h2><div class="controls"><label>Attempt <select id="run"></select></label><button id="prev">Previous</button><button id="next">Next</button><span id="step"></span></div><input id="slider" type="range" min="0" value="0"><div id="caption" class="muted"></div><img id="frame" alt="Approved reference and current candidate actually sent to the agent"><pre id="summary"></pre><details><summary>Actual action</summary><pre id="action"></pre></details><details><summary>Saved candidate source</summary><pre id="source"></pre></details>
 </main><script>const data=DATA;const $=id=>document.getElementById(id);$('accepted').textContent=data.accepted+'/200';$('adaptive').textContent=(100*data.study.arms.adaptive.raw_mixed_share).toFixed(1)+'%';$('fixed').textContent=(100*data.study.arms.fixed.raw_mixed_share).toFixed(1)+'%';
 const groups={};for(const run of data.runs){let r=run.result;if(r.stage!=='private_evaluation')continue;let k=r.model+'|'+r.task_id;(groups[k]??=[]).push(r)}
 for(const [key,rows] of Object.entries(groups)){let done=rows.filter(r=>r.status==='completed'),tr=document.createElement('tr');let partial=done.length?done.reduce((a,r)=>a+(r.final_result?r.final_result.passed/r.final_result.total:0),0)/done.length:null;for(const value of [key.split('|')[0],key.split('|')[1],done.filter(r=>r.success).length+'/'+done.length,partial===null?'—':(100*partial).toFixed(1)+'%',rows.length-done.length]){let td=document.createElement('td');td.textContent=value;tr.append(td)}$('scores').append(tr)}
 for(let i=0;i<data.runs.length;i++){let o=document.createElement('option');o.value=i;o.textContent=data.runs[i].id+' · '+data.runs[i].result.status;$('run').append(o)}
 let current=0;function render(){let r=data.runs[Number($('run').value)];if(!r)return;current=Math.max(0,Math.min(current,r.steps.length-1));let s=r.steps[current];$('slider').max=r.steps.length-1;$('slider').value=current;$('step').textContent='Step '+s.step;$('caption').textContent='Reward '+s.reward.toFixed(6)+' · '+s.remaining+' actions remain';$('frame').src='../'+s.image;$('summary').textContent=s.summary;$('action').textContent=JSON.stringify(s.action,null,2);$('source').textContent=s.source}
 $('run').onchange=()=>{current=0;render()};$('slider').oninput=e=>{current=Number(e.target.value);render()};$('prev').onclick=()=>{current--;render()};$('next').onclick=()=>{current++;render()};render();</script></html>'''.replace('DATA',payload)
 (out/'index.html').write_text(html)
 print(json.dumps({'accepted':len(pool['records']),'replay_attempts':len(replay),'output':str(out)}))
if __name__=='__main__':main()
