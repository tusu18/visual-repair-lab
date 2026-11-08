# Durable concurrent episodes

The v7 store uses a new SQLite schema and runtime/bank seal. Keep an old database unchanged and select a new filename for v7. SQLite uses WAL, full synchronization and brief commits. Use a local disk; network filesystem locking semantics are not certified.

1. Read the episode checkpoint and expected step, then close the database connection.
2. A bounded private worker restores the data-only checkpoint. It reconstructs trusted reference callables from the sealed specification, restores RNG/state/credit ledgers and cached progress, then executes **one** new action.
3. Rendering and code evaluation run outside the database write transaction. Independent episodes can execute concurrently.
4. Begin a short write transaction, recheck request ID and expected step, then atomically save the new checkpoint, public response, action history and idempotency receipt.
5. A competing same-episode action gets HTTP409; it must resume. An identical request ID gets the original response, including after a restart. Two concurrent identical calls may compute twice, but only one transition commits.

Snapshots contain only tagged JSON data, compressed with zlib, with an integrity digest; no pickle, executable objects or actor-supplied snapshot/path is accepted by HTTP. They remain private evaluator state. Reference functions and spaces are reconstructed rather than serialized. RGB frames are regenerated; screenshots match the direct environment exactly in checkpoint tests. The digest detects accidental corruption; it is not protection against an administrator who can rewrite both database and code.

Worker timeouts are unscored and leave the old checkpoint untouched. Retrying the same **saved decision** does not consume an extra action. A fresh worker process bounds a stuck interpreter; process startup still has overhead. Candidate programs themselves replay public event history as required by the task contract, but the server no longer re-executes old actions, edits, gradings or reward calculations. Serialization and audit-history size still grow with episode length.

HTTP backlog is 64. The operator benchmark supports up to eight concurrent episodes. Actual throughput depends on CPU, rendering and candidate complexity. The included concurrency report measures evaluator wall time, not model inference time or simulator ticks. A multi-host evaluator, distributed scheduling, production authentication and an adversarial OS sandbox are outside this release.

## Actor API

All POST bodies are JSON; request bodies are bounded to 100 KB.

- `/episodes`: `{"task_id":"train-0000","evidence":"image","mode":"evaluation"}`.
- `/resume`: `{"episode_id":"..."}`.
- `/step`: `{"episode_id":"...","request_id":"a-stable-unique-ID","expected_step":0,"action":{"type":"observe"}}`.

Persist the request ID and completed model decision **before** sending a step. On transport failure, retry those bytes. On HTTP409, resume and resolve the conflict; never generate another decision merely to replace a saved completion. Public responses contain context/image/reward/current step, with final grade after submit. Private snapshots, references and grading cases never enter these responses.

Episode IDs are random bearer capabilities. Bind to loopback or a private container network, do not publish the evaluator on the Internet. The GUI's benchmark routes are operator controls and must not be agent tools. Cross-origin browser POSTs are rejected. TLS/authentication should be added by the deploying lab if remote operator access is required.
