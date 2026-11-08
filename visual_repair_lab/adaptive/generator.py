"""Seeded train-only proposal generator, mutation editor and rejection ledger."""
import copy
import random
from collections import Counter
from .dsl import BASE_MASKS,CONDITIONS,FAMILIES,HOOKS,VALUES,seal
from .certify import certify


def propose(rng,index):
    family=FAMILIES[index%len(FAMILIES)]
    count=rng.choice((1,1,2,2,3))
    operations=rng.sample(list(HOOKS[family]),count)
    rules=[dict(op=op,hook=HOOKS[family][op],condition=rng.choice(CONDITIONS[family]),value=rng.choice(VALUES[family][op])) for op in operations]
    return dict(id=f'train-{index:04}',family=family,world={'mask':rng.choice(BASE_MASKS)},rules=rules,
                budget=rng.choice((12,16,24,32,48)),starting_point=rng.choice(('baseline','baseline','proposal','starter')),
                split='train',scene_seed=3411000+index)


def evaluate_proposal(spec):
    try:receipt=certify(spec)
    except Exception as exc:receipt={'accepted':False,'reasons':['evaluator:'+type(exc).__name__]}
    return {'spec':spec,'certificate':receipt}


def filter_records(records,existing=()):
    seen={r['certificate']['behavior_fingerprint'] for r in existing if r['certificate']['accepted']}
    for row in records:
        c=row['certificate']
        if c['accepted']:
            if c['behavior_fingerprint'] in seen:
                c['accepted']=False;c['reasons'].append('duplicate_behavior')
            else:seen.add(c['behavior_fingerprint'])
    reasons=Counter(reason for r in records for reason in r['certificate']['reasons'])
    return {'records':[r for r in records if r['certificate']['accepted']],
            'rejected':[r for r in records if not r['certificate']['accepted']],
            'rejection_counts':dict(reasons),'proposed':len(records)}


def edit_task(spec,p,rng,serial):
    child=copy.deepcopy(spec);kind=None
    if p>=.75:
        available=[op for op in HOOKS[spec['family']] if op not in {r['op'] for r in spec['rules']}]
        if len(child['rules'])<3 and available and rng.random()<.65:
            op=rng.choice(available);child['rules'].append(dict(op=op,hook=HOOKS[spec['family']][op],
                condition=rng.choice(CONDITIONS[spec['family']]),value=rng.choice(VALUES[spec['family']][op])))
            kind='add_rule'
        else:child['budget']=max(8,child['budget']-4);kind='cut_budget'
    elif p==0:
        if len(child['rules'])>1 and rng.random()<.65:
            child['rules'].pop(rng.randrange(len(child['rules'])));kind='drop_rule'
        else:child['budget']=min(48,child['budget']+4);kind='add_budget'
    else:return None
    if child==spec:return None
    child['id']=f"edit-{serial:05}"
    child['scene_seed']=3420000+serial
    return child,{'parent_id':spec['id'],'operation':kind,'parent_p':p,'parent_sha256':seal(spec)}
