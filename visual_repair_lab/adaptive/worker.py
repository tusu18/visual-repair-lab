"""Bounded server subprocess: public observations only on stdout."""
import base64,io,json,sys
from PIL import Image
from .factory import make_env


def evaluate(request):
    env=make_env(request['spec'],reward_mode=request['mode'],evidence=request['evidence']);observation,_=env.reset()
    reward=0.;terminated=truncated=False
    for action in request['actions']:observation,reward,terminated,truncated,_=env.step(action)
    data=io.BytesIO();Image.fromarray(observation['image']).save(data,format='PNG')
    context=json.loads(observation['context'])
    if context.get('final_result'):
        context['final_result']={k:context['final_result'][k] for k in ('success','passed','total','final_revision_activated','chosen_verification_tests')}
    return {'context':context,'image_png_base64':base64.b64encode(data.getvalue()).decode(),
            'reward':reward,'terminated':terminated,'truncated':truncated,'step':env.steps}


if __name__=='__main__':
    try:print(json.dumps({'ok':True,'result':evaluate(json.load(sys.stdin))},allow_nan=False))
    except Exception as exc:
        print(json.dumps({'ok':False,'error':type(exc).__name__}));sys.exit(1)
