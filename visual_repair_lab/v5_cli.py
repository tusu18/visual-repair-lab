"""V5 delivery commands. Hosted model generation is always explicit."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Visual Repair Lab v5 trusted evaluator')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('serve');s.add_argument('--bank',required=True);s.add_argument('--database',required=True)
    s.add_argument('--host',default='127.0.0.1');s.add_argument('--port',type=int,default=8810);s.add_argument('--timeout',type=float,default=45)
    c=sub.add_parser('certify');c.add_argument('--proposals',required=True);c.add_argument('--output',required=True)
    g=sub.add_parser('generate');g.add_argument('--prompt',required=True);g.add_argument('--schema',required=True)
    g.add_argument('--output-dir',required=True);g.add_argument('--model',default='gpt-6.1-sol');g.add_argument('--effort',default='high')
    args=p.parse_args()
    if args.command=='serve':
        from .v5.server import serve
        serve(args.database,args.bank,args.host,args.port,args.timeout)
    elif args.command=='certify':
        from .v5.generator import release_bank
        b=release_bank(json.loads(Path(args.proposals).read_text()),args.output)
        print(json.dumps({'accepted':len(b['records']),'rejected':len(b['rejected']),'bank_sha256':b['bank_sha256']}))
    else:
        from .v5.codex_transport import DiscoveryCodexPolicy
        out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
        prompt=Path(args.prompt).read_text();schema=json.loads(Path(args.schema).read_text())
        (out/'prompt.txt').write_text(prompt);(out/'schema.json').write_text(json.dumps(schema,indent=2))
        result=DiscoveryCodexPolicy(model=args.model,effort=args.effort,timeout=240).call(prompt,[],schema)
        (out/'response.json').write_text(json.dumps(result,indent=2))
        if result['status']!='ok':
            raise SystemExit('Generator interrupted; prefix preserved. No tasks accepted.')
        data=json.loads(result['completion_text']);proposals=[]
        for row in data['proposals']:
            row=dict(row);row.pop('rationale',None);row['rules']=json.loads(row.pop('rules_json'));proposals.append(row)
        (out/'proposals.json').write_text(json.dumps(proposals,indent=2))
        print('Proposals saved; run certify before release. No generated code is executed.')
