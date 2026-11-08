import concurrent.futures,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'actor_client'))
from hosted_transport import SpendLedger
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_final_v6 import public_history


def test_shared_cap_reservations_cannot_race_above_six_dollars(tmp_path):
 ledger=SpendLedger(tmp_path/'budget.db')
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  results=list(pool.map(lambda _:ledger.reserve('anthropic/test',.8),range(20)))
 assert sum(x is not None for x in results)==7
 assert float(ledger.summary()['unresolved_upper_bound'])<=6


def test_unknown_charges_stay_reserved_and_actual_cost_releases_remainder(tmp_path):
 ledger=SpendLedger(tmp_path/'budget.db');key=ledger.reserve('anthropic/test',3)
 ledger.settle(key,None,'unknown');assert ledger.reserve('anthropic/test',4) is None
 ledger.settle(key,.1,'completed');assert ledger.reserve('anthropic/test',4) is not None


def test_public_history_does_not_repeat_old_code_or_add_hidden_information():
 rows=[{'step':1,'action':{'type':'edit','source':'def solve(state):\n    return {}\n'},'summary':'Public edit'}]
 compact=public_history(rows)
 assert 'source' not in compact[0]['action'] and compact[0]['summary']=='Public edit'
 assert rows[0]['action']['source'].startswith('def solve')


def test_prompt_preserves_active_code_and_binds_only_current_image():
 from benchmark_final_v6 import prompt_for
 public={'step':2,'context':{'source':'saved','source_revision':2,'active_revision':1,'observation_id':'public-id','scene':{'x':4},'family':'drag_hysteresis','budget':24,'remaining':22}}
 prompt=prompt_for(public,[],active_source='earlier running code')
 assert 'earlier running code' in prompt and 'CURRENT observation' in prompt and 'image_index":0' in prompt
 assert 'Earlier images are not reattached' in prompt and 'Current saved source:\nsaved' in prompt
