# v7 delivery validation

This release fixes two demonstrated delivery blockers: reward for submitting inherited correct behavior, and a global write lock held during full-history replay. It also supplies a local operator GUI and VLM benchmark workflow. Earlier versions and model grades are untouched. No real-model calls or training were run for this revision.

## Reward audit

All **134/134 bundled tasks** were reset and submitted unchanged in train mode. Each has zero terminal reward and −0.025 total return for the one action. Reapplying the old formula to the same starting grades reproduces **89/134 positive unchanged returns**, median **+1.2411**; the new median is **−0.025**. The full per-task receipt is `validation/unchanged-submit-audit.json`.

The previously worst case, `train-0116`, starts at103/109 protected checks and still scores103/109 unchanged. Its v7 terminal reward is0 and total return−0.025. The GUI screenshot and fresh installed-wheel smoke verify this through the official interface.

A deliberately incomplete brush patch on `train-0001` still passes zero whole protected cases, but earns **0.31293** training progress for repaired output fields. This is a scripted regression fixture, not model ability or proof of reward alignment. Test-power reward is now at most0.1 per action, within the0.5 episode cap. Discounted checkpoint priors are scheduling-only and remain separate from current measured outcomes.

## Server throughput

Same local harness, eight interaction steps per episode, including creation, process startup and rendering:

| Path | One episode | Eight concurrent episodes | Accepted concurrent steps | Worst concurrent step |
|---|---:|---:|---:|---:|
| Legacy v6 replay + write lock |9.04s|72.24s|64/64|18.76s|
| v7 incremental checkpoint |3.26s|5.84s|64/64|0.49s|
| Final v7 safeguard confirmation |3.37s|6.38s|64/64|0.65s|

All listed runs have zero worker failures. These are local CPU microbenchmarks; load and caches are not controlled, and they do not include model inference or establish a production latency guarantee. `validation/concurrency-benchmark.json` preserves the legacy comparison; `validation/final-concurrency/` preserves the final confirmation.

An additional HTTP smoke on the installed wheel completed24 scripted benchmark attempts across three families at concurrency8, then8 simultaneous HTTP episodes with idempotent duplicate step requests. All completed with no interruptions and unchanged-submit return−0.025. These scripted attempts are never reported as VLM results.

## Tests and installation

- **307/307 package tests pass on macOS arm64 / Python3.13.0**,134.9s, bounded420s runner.
- **307/307 package tests pass in clean Linux arm64 / Python3.12.16**,187.7s, bounded420s runner.
- All275 existing tests are retained. Only distribution-version expectations were bumped to7.0.0 in three existing assertions;32 new cases cover relative rewards, comment edits, zero-whole-case field progress, exact checkpoint images/context/rewards/RNG, private detection, corruption rejection, restart/idempotency, timeout recovery, independent-episode concurrent writes, same-episode races, decision preservation and aged-prior separation.
- The initial full macOS run failed three stale version assertions, with300 passing tests; its receipt is preserved. Later successful full runs do not overwrite it.
- Fresh empty virtual environment, pinned runtime dependencies and wheel-only installation outside the source tree pass dependency, CLI, GUI-asset, installed-source parity and official-store smoke checks. The GUI running on port8822 uses that installed wheel.
- Both isolated legacy core modules and the independent references/graders remain byte-identical to v6. Existing model scores are not recomputed. Runtime and bank seals, fresh databases and actor-only public responses remain enforced.

The ZIP includes source, wheel, source archive, dependency locks, Dockerfiles, compose, operator guide, reward/server contracts, validation receipts and source checksums. The Dockerfile pins its Python base digest and uses a nonroot evaluator. Architecture-specific container testing covers Linux arm64; x86/GPU model serving is not certified by these receipts.

## Scope of readiness

Ready for **local research handoff and user-operated model benchmarks**, within the documented actor boundary. It is not proof of universal grader non-gameability, production sandbox security, broad difficulty calibration or training benefit. The public134-task bank is development-used. New checkpoint priors do not create fresh outcomes, and field-level reward alignment still needs a real training/evaluation experiment.
