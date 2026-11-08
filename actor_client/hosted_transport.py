"""Pinned OpenRouter calls with durable conservative spending reservations.

Keys are held only by this Python process; never included in payload/evidence.
"""
import base64,hashlib,json,re,sqlite3,time,urllib.request,urllib.error,uuid
from decimal import Decimal
from pathlib import Path


class SpendLedger:
    def __init__(self,path,cap='6.00'):
        self.path=str(path);self.cap=Decimal(cap)
        with self.connect() as db:db.execute('CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,model TEXT,reserved TEXT,cost TEXT,status TEXT)')
    def connect(self):return sqlite3.connect(self.path,timeout=20)
    def reserve(self,model,amount):
        amount=Decimal(str(amount))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');rows=db.execute('SELECT model,reserved,cost FROM calls').fetchall()
            used=sum((Decimal(c if c is not None else r) for _,r,c in rows),Decimal(0))
            q_used=sum((Decimal(c if c is not None else r) for m,r,c in rows if m.startswith('qwen/')),Decimal(0))
            if used+amount>self.cap or model.startswith('qwen/') and q_used+amount>Decimal('2.00'):return None
            token=uuid.uuid4().hex;db.execute('INSERT INTO calls VALUES (?,?,?,?,?)',(token,model,str(amount),None,'reserved'));return token
    def settle(self,token,cost,status):
        with self.connect() as db:db.execute('UPDATE calls SET cost=?,status=? WHERE id=?',(None if cost is None else str(cost),status,token))
    def summary(self):
        with self.connect() as db:rows=db.execute('SELECT model,reserved,cost,status FROM calls').fetchall()
        return {'cap':str(self.cap),'known_cost':str(sum((Decimal(c) for _,_,c,_ in rows if c is not None),Decimal(0))),
                'unresolved_upper_bound':str(sum((Decimal(r) for _,r,c,_ in rows if c is None),Decimal(0))),'calls':len(rows)}


class HostedPolicy:
    def __init__(self,key,profile,ledger):self.key=key;self.profile=profile;self.ledger=ledger
    def call(self,prompt,images,schema):
        p=self.profile;marker='\nCURRENT_PUBLIC_STATE_AND_HISTORY:\n'
        if marker in prompt:
            prefix,suffix=prompt.split(marker,1);block={'type':'text','text':prefix}
            if p['model'].startswith('anthropic/'):block['cache_control']={'type':'ephemeral'}
            content=[block,{'type':'text','text':marker+suffix}]
        else:content=[{'type':'text','text':prompt}]
        hashes={}
        for i,path in enumerate(images):
            data=Path(path).read_bytes();hashes[f'image_{i:02}.png']=hashlib.sha256(data).hexdigest()
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode(),'detail':'high'}})
        payload={'model':p['model'],'messages':[{'role':'user','content':content}],
                 'response_format':{'type':'json_schema','json_schema':{'name':'environment_action','strict':True,'schema':schema}},
                 'max_tokens':4096,'reasoning':{'effort':'low'},'stream':False,
                 'provider':{'only':[p['provider']],'allow_fallbacks':False,'require_parameters':True,
                    'max_price':{'prompt':p['prompt_per_million'],'completion':p['completion_per_million']}}}
        # One text token cannot exceed its UTF-8 byte count. Image allowance is
        # deliberately above these 1200x700 frames. 2x input margin covers writes.
        upper=2*(len(prompt.encode())+8192*len(images))*p['prompt_per_million']/1e6+4096*p['completion_per_million']/1e6
        token=self.ledger.reserve(p['model'],upper)
        if token is None:return {'status':'budget_censored','completion_text':'','audit':{'valid':False,'tool_calls':0},'upper_reservation':upper}
        wire=json.dumps(payload,separators=(',',':')).encode();start=time.monotonic()
        request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=wire,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'VisualRepairLab/6.0','X-OpenRouter-Title':'Visual Repair Lab benchmark'})
        try:
            with urllib.request.urlopen(request,timeout=180) as response:raw=json.load(response)
        except urllib.error.HTTPError as exc:
            # No raw SDK exceptions, headers or credentials enter evidence.
            detail=re.sub(r'sk[-_][A-Za-z0-9_-]+','[REDACTED]',exc.read(4000).decode(errors='replace').replace(self.key,'[REDACTED]'))
            self.ledger.settle(token,None,'http_error_unknown_charge')
            return {'status':'provider_interruption','http_status':exc.code,'error_detail':detail,'completion_text':'','reservation_id':token,'audit':{'valid':False,'tool_calls':0},'wall_seconds':time.monotonic()-start}
        except (OSError,TimeoutError,ValueError):
            self.ledger.settle(token,None,'transport_unknown_charge')
            return {'status':'transport_interruption','completion_text':'','reservation_id':token,'audit':{'valid':False,'tool_calls':0},'wall_seconds':time.monotonic()-start}
        raw=json.loads(json.dumps(raw).replace(self.key,'[REDACTED]'))
        cost=raw.get('usage',{}).get('cost');self.ledger.settle(token,cost,'completed' if cost is not None else 'unknown_charge')
        choices=raw.get('choices',[]);message=choices[0].get('message',{}) if choices else {};text=message.get('content') or ''
        if isinstance(text,list):text=''.join(x.get('text','') for x in text if isinstance(x,dict))
        tools=message.get('tool_calls',[])
        return {'status':'ok' if choices else 'provider_interruption','completion_text':text,'raw_response':raw,
                'usage':raw.get('usage',{}),'reservation_id':token,'request_sha256':hashlib.sha256(wire).hexdigest(),
                'requested_model':p['model'],'requested_provider':p['provider'],'requested_parameters':{k:v for k,v in payload.items() if k!='messages'},
                'image_sha256':hashes,'image_count':len(images),'finish_reason':choices[0].get('finish_reason') if choices else None,
                'audit':{'valid':not tools and bool(choices),'tool_calls':len(tools)},'wall_seconds':time.monotonic()-start}
