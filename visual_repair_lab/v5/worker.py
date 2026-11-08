"""One bounded evaluator process. stdin/stdout are trusted server-only channels."""
import base64
import io
import json
import sys
from PIL import Image
from .env import GeneratedRepairEnv


def evaluate(request):
    env = GeneratedRepairEnv(request['spec'], reward_mode=request['mode'], evidence=request['evidence'])
    observation, _ = env.reset()
    reward, terminated, truncated = 0., False, False
    for action in request['actions']:
        observation, reward, terminated, truncated, _ = env.step(action)
    data = io.BytesIO()
    Image.fromarray(observation['image']).save(data, format='PNG')
    context = json.loads(observation['context'])
    # Case-by-case protected results and evaluator metrics never cross the actor boundary.
    if context.get('final_result'):
        context['final_result'] = {k: context['final_result'][k] for k in
            ('success', 'passed', 'total', 'final_revision_activated', 'chosen_verification_tests')}
    return {'context': context, 'image_png_base64': base64.b64encode(data.getvalue()).decode(),
            'reward': reward, 'terminated': terminated, 'truncated': truncated, 'step': env.steps}


if __name__ == '__main__':
    try:
        print(json.dumps({'ok': True, 'result': evaluate(json.load(sys.stdin))}, allow_nan=False))
    except Exception as exc:
        # Never expose interpreter internals or a private world in actor errors.
        print(json.dumps({'ok': False, 'error': type(exc).__name__}))
        sys.exit(1)
