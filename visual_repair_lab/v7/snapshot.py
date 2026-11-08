"""Private data-only checkpoints. No pickle, imports, functions or actor paths."""
import base64
import json
import zlib
import numpy as np
from vision_sort.visual_training.mutants import compile_trusted
from visual_repair_lab.adaptive.compiler import compile_source
from visual_repair_lab.adaptive.dsl import seal
from .env import make_env

EXCLUDED = {'action_space','observation_space','_gold','_rule_bugs','_starter_solve',
            '_mutants','_heldout_mutants','solve_gold','_image','_reference','_candidate'}


def encode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, np.random.Generator):
        return {'t':'rng','v':encode(value.bit_generator.state)}
    if isinstance(value, np.ndarray):
        return {'t':'array','dtype':str(value.dtype),'v':value.tolist()}
    if type(value) in (dict,list,tuple,set):
        name = type(value).__name__
        items = [[encode(k),encode(v)] for k,v in value.items()] if name=='dict' else [encode(x) for x in value]
        if name=='set':
            items.sort(key=lambda x:json.dumps(x,sort_keys=True))
        return {'t':name,'v':items}
    raise TypeError('Unsupported checkpoint field: '+type(value).__name__)


def decode(value):
    if not isinstance(value, dict):
        return value
    t, items = value['t'], value['v']
    if t=='dict': return {decode(k):decode(v) for k,v in items}
    if t=='list': return [decode(x) for x in items]
    if t=='set': return {decode(x) for x in items}
    if t=='tuple': return tuple(decode(x) for x in items)
    if t=='array': return np.array(items,dtype=value['dtype'])
    if t=='rng':
        rng=np.random.default_rng();rng.bit_generator.state=decode(items);return rng
    raise ValueError('Unknown checkpoint type')


def dump(env):
    payload={'version':1,'fields':encode({k:v for k,v in vars(env).items() if k not in EXCLUDED})}
    wire=json.dumps(payload,allow_nan=False,separators=(',',':')).encode()
    return {'sha256':seal(payload),'data':base64.b64encode(zlib.compress(wire)).decode()}


def restore(config, snapshot):
    payload=json.loads(zlib.decompress(base64.b64decode(snapshot['data'])))
    if payload['version']!=1 or snapshot['sha256']!=seal(payload):
        raise ValueError('Checkpoint integrity mismatch')
    env=make_env(config['spec'],reward_mode=config['mode'],evidence=config['evidence'])
    vars(env).update(decode(payload['fields']))
    if env.family!='detection_overlay':
        env._starter_solve=compile_trusted(env.starter)
        env._mutants={f'rule_{i}':compile_trusted(compile_source(config['spec'],disabled=[i]))
                      for i in range(len(config['spec']['rules']))}
        env._heldout_mutants={}
        env._refresh()
    return env
