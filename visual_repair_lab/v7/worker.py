"""One new transition per checkpoint; old actions are never replayed here."""
import base64
import io
import json
import sys
from PIL import Image
from .env import make_env
from .snapshot import dump, restore


def evaluate(request):
    config=request['config']
    if request.get('snapshot') is None:
        env=make_env(config['spec'],reward_mode=config['mode'],evidence=config['evidence'])
        observation,_=env.reset()
    else:
        env=restore(config,request['snapshot'])
        observation,_,_,_,_=env.step(request['action'])
    data=io.BytesIO();Image.fromarray(observation['image']).save(data,format='PNG')
    context=json.loads(observation['context'])
    if context.get('final_result'):
        context['final_result']={k:context['final_result'][k] for k in
            ('success','passed','total','final_revision_activated','chosen_verification_tests')}
    public={'context':context,'image_png_base64':base64.b64encode(data.getvalue()).decode(),
        'reward':env.reward,'terminated':env.terminated,'truncated':env.truncated,'step':env.steps,
        'total_return':env.task_return+env.process_return}
    if env.reward_ledger:
        components=env.reward_ledger[-1]['components']
        public['reward_components']={k:v for k,v in components.items()
            if env.mode=='train' or k in ('action','invalid','terminal','budget_exhaustion','unactivated_submission')}
    if env.terminated:
        public['repair_summary']={'initial_passed':env._initial_passed,'total':env._initial_total,
            'final_passed':env.final_result['passed'],
            'net_improvement':env.final_result['passed']-env._initial_passed}
    return {'ok':True,'result':public,'checkpoint':dump(env)}


if __name__=='__main__':
    try:
        print(json.dumps(evaluate(json.load(sys.stdin)),allow_nan=False))
    except Exception:
        print(json.dumps({'ok':False,'error':'evaluator_interruption','scored':False,'retryable':True}))
        sys.exit(1)
