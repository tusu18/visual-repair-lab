"""Target-independent ideal policy over the complete same-family train corpus.

This second gate removes the per-task operation-palette hint from the first gate.
The ideal prior is the same whole program catalogue for every task in a family.
"""
import copy
from collections import defaultdict
from vision_sort.visual_training.mutants import compile_trusted
from visual_repair_lab.v5.compare import equivalent
from .certify import decision_tree,visible_signature,batch
from .compiler import compile_source
from .dsl import seal
from .probes import experiments,state_for,grading_states


def certify_against_catalogue(record,catalogue):
    spec=record['spec'];family=spec['family']
    sources=[compile_source(r['spec']) for r in catalogue if r['spec']['family']==family]
    programs=[compile_trusted(source) for source in sources]
    probes=list(experiments(family))
    if family=='drag_hysteresis':
        # Hysteresis cannot be identified by single resizes from the initial
        # state. Both directions and prior layout states must be challenged.
        for before in (440,640):
            for after in (459,461,479,481,519,521,539,541):
                probes.append([{'kind':'resize','width':before},{'kind':'resize','width':after}])
    sequences=[[]]+[[{'kind':'reset_scene'}]+s for s in probes]
    probe_states=[state_for(family,spec['scene_seed'],s) for s in sequences]
    matrix=[[visible_signature(fn(copy.deepcopy(s))) for fn in programs] for s in probe_states]
    aliases=defaultdict(list)
    for i in range(len(programs)):aliases[tuple(row[i] for row in matrix)].append(i)
    representatives=[];ambiguous=False;final=grading_states(spec,'final')
    for group in aliases.values():
        first=group[0];representatives.append(first)
        if len(group)>1:
            values=[programs[first](copy.deepcopy(s)) for s in final]
            for other in group[1:]:
                if any(not equivalent(programs[other](copy.deepcopy(s)),v) for s,v in zip(final,values)):ambiguous=True
    receipt={'catalogue_sha256':seal([r['certificate']['source_sha256'] for r in catalogue if r['spec']['family']==family]),
             'catalogue_programs':len(programs),'operation_palette_hint':False,'rounding_or_probe_alias':ambiguous}
    if ambiguous:return dict(receipt,accepted=False,reasons=['whole_catalogue_alias'])
    try:tree=decision_tree([[row[i] for i in representatives] for row in matrix],[len(s) for s in sequences])
    except ValueError:return dict(receipt,accepted=False,reasons=['whole_catalogue_unidentifiable'])
    # Use ONE target-independent tree, but charge this world's actual branch.
    # A short branch must not pay for an unrelated world's longest branch.
    target=compile_trusted(compile_source(spec));node=tree;history=[]
    while 'leaf' not in node:
        history+=copy.deepcopy(sequences[node['experiment']])
        signature=visible_signature(target(state_for(family,spec['scene_seed'],history)))
        if signature not in node['branches']:return dict(receipt,accepted=False,reasons=['ideal_trajectory_observation_mismatch'])
        node=node['branches'][signature]
    source=sources[representatives[node['leaf']]]
    from .reference import reference
    from vision_sort.visual_training import tasks
    checked=batch(source,final)
    valid=all(r['ok'] and tasks.valid(family,s,r['result']) and equivalent(r['result'],reference(spec,s)) for s,r in zip(final,checked))
    current=state_for(family,spec['scene_seed'],history);current_run=batch(source,[current])[0]
    valid=valid and current_run['ok'] and equivalent(current_run['result'],target(copy.deepcopy(current)))
    total=len(history)+3;reasons=[]
    if not valid:reasons.append('ideal_executed_repair_failed')
    if total+2>spec['budget']:reasons.append('whole_catalogue_budget')
    return dict(receipt,accepted=not reasons,reasons=reasons,executed_ideal_valid=valid,ideal_worst_total_with_repair=tree['cost']+3,
                ideal_total_with_repair=total,spare_actions=spec['budget']-total,
                private_tree=tree,private_representatives=representatives,private_sequences=sequences)
