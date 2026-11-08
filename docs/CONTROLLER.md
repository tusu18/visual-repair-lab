# Trainer integration

The trainer owns the task pool and the model; the actor owns neither the pool specification nor the evaluator filesystem.

```python
from visual_repair_lab.adaptive.teacher import AdaptiveTeacher
teacher = AdaptiveTeacher('curriculum.db', policy_id='my-checkpoint-sha256', seed=7)
# Add initial records from the certified TRAIN bank using teacher.curriculum.add.
record, reason = teacher.next_task()
# Install record privately in the evaluator; give the actor only public observations.
# Run eight independent attempts with the same frozen policy settings.
summary = teacher.observe_group(record['spec']['id'], round_index=1, attempts=attempts)
```

An attempt record includes a unique attempt_id, status, binary full-repair success and optional return. Incomplete/transport-interrupted groups are saved but do not produce p. Do not label provider errors as model failures. Never regenerate a successful model decision while recovering a failed transport call.

The controller uses the latest complete eight-attempt group for the current policy checkpoint. Active weights are `p*(1-p)`. At p=0 it parks the task; above0.9 it retires it. With eight samples, retirement normally means8/8. A small sample is uncertain; the reported Wilson interval is descriptive, not proof of intrinsic task difficulty.

When active and unseen tasks coexist, at most a20% branch probability is devoted to unseen calibration; active tasks retain priority. Retired tasks have a5% replay branch. An all-retired pool triggers hardening/generation, not unlimited easy replay. All-zero pools trigger softening/new proposals. A bounded generation retry limit reports failure to find an admissible task instead of looping forever.

At p>=0.75 the editor adds a rule or removes four budget actions. At p=0 it drops a rule or adds four. Every child is checked against the current complete catalogue plus that child. The certificate must leave two actions spare. Pool mutations and random state are persisted; teacher task IDs include the policy identity. New policy IDs invalidate estimates while preserving old groups and tasks.

Do not import teacher.next_task as an actor tool. It returns private rules and certificates to the trusted trainer. The public CLI `next` prints only a task ID and selection reason. Actor access is HTTP create/resume/step with persistent request IDs. A changed bank/runtime requires a new evaluator database; never silently resume under changed task rules. The curriculum also seals its generator/runtime: use a new curriculum database after changing that source. Old databases and all outcomes remain preserved.

The binary DAPO-style keep flag is separate from dense-reward variance. All-fail groups can still contain useful partial-repair differences, and all-pass groups can still differ in action cost. Choose the training objective deliberately; the binary mixed-share headline is a proxy, not a direct gradient measurement.
