"""Explicit offline generation and isolated evaluator commands."""
import argparse,json,random
from pathlib import Path
from .dsl import seal


def main():
    p=argparse.ArgumentParser(description='Visual Repair Lab 6 adaptive research environment')
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--output',required=True);g.add_argument('--count',type=int,default=200);g.add_argument('--seed',type=int,default=7)
    s=sub.add_parser('serve');s.add_argument('--bank',required=True);s.add_argument('--database',required=True);s.add_argument('--port',type=int,default=8813)
    s.add_argument('--host',default='127.0.0.1');s.add_argument('--timeout',type=float,default=90)
    n=sub.add_parser('next');n.add_argument('--database',required=True);n.add_argument('--policy-id',required=True)
    u=sub.add_parser('record-group');u.add_argument('--database',required=True);u.add_argument('--policy-id',required=True);u.add_argument('--task-id',required=True)
    u.add_argument('--round',type=int,required=True);u.add_argument('--attempts',required=True)
    args=p.parse_args()
    if args.command=='serve':
        from .server import serve
        serve(args.database,args.bank,args.host,args.port,args.timeout)
    elif args.command=='generate':
        from .generator import propose,evaluate_proposal,filter_records
        rng=random.Random(args.seed);rows=[evaluate_proposal(propose(rng,i)) for i in range(args.count)]
        result=filter_records(rows);result['bank_sha256']=seal(result['records'])
        from .pool_gate import certify_against_catalogue
        provisional=list(result['records']);accepted=[]
        for record in provisional:
            certificate=certify_against_catalogue(record,provisional)
            record['certificate']['whole_catalogue_gate']=certificate
            if certificate['accepted']:accepted.append(record)
            else:
                record['certificate']['accepted']=False;record['certificate']['reasons']+=certificate['reasons'];result['rejected'].append(record)
        result['records']=accepted;result['bank_sha256']=seal(accepted)
        from collections import Counter
        result['rejection_counts']=dict(Counter(reason for r in result['rejected'] for reason in r['certificate']['reasons']))
        Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps({'accepted':len(result['records']),'rejected':len(result['rejected'])}))
    else:
        from .curriculum import Curriculum
        c=Curriculum(args.database,args.policy_id)
        if args.command=='next':
            record,reason=c.choose();result={'task_id':record['spec']['id'] if record else None,'selection':reason}
        else:result=c.record(args.task_id,args.round,json.loads(Path(args.attempts).read_text()))
        print(json.dumps(result))

if __name__=='__main__':main()
