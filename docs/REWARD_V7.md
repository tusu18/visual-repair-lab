# Reward specification · v7

Let S be the starting code's protected passed-check count, F the submitted code's count, and N the total. These are computed with the same independent reference, schema checks and raw numeric tolerance (absolute 1e-6, relative 1e-9). S is private during the episode and immutable across edits and restarts. Final grading still reports F/N and succeeds only when F=N.

Terminal reward:

- F=S: **0**.
- F>S: **8 × (F−S)/(N−S)**, plus **2** if F=N.
- F<S: **−2 × (S−F)/N**.
- S=N: preserving the already correct start earns **0**, and regression is penalized. These audit episodes are not repair-improvement opportunities and should be stratified in training.

Example: starting 60/100, final 80/100 pays 4. Starting 60/100, final 100/100 pays 10. Starting 60/100, final 60/100 pays zero, including comment edits. Costs still apply: one unchanged submit returns −0.025.

Every action costs 0.025; invalid actions incur another 0.1. Unactivated submission costs another 0.25. Exhausting the action budget without submit costs 2. Submitting saved code without activation remains allowed and is accurately reported; no test or submission is invented on the agent's behalf.

## Training-only process feedback

Evaluation mode applies only terminal rewards and costs. Train mode, restricted to train tasks, also applies the following bounded signals. The independent final grader is unchanged.

- **Repair progress:** up to 0.75 per episode for increasing schema-valid field agreement above the starting value and the previous high-water mark, measured on disjoint progress states. Agreement recursively averages expected dictionary fields/list entries with raw tolerance; invalid whole outputs score zero. It can distinguish partial repairs when every whole case still fails. Later losses cost 0.75 times the decrease from the previous field score. Reverting code cannot repay earlier progress.
- **Test power:** at most 0.1 per action and 0.5 per episode, after a behavior-changing active edit. Omission witnesses remain credited once; excess per-action credit is discarded, not deferred for repeating a test. A one-rule task may therefore have at most 0.1 total test-power credit. The metric records structural exposure, not attention or causal understanding.
- **Verification:** existing revision-bound cap 0.25.
- **Accurate visual comparison:** existing cap 0.25.
- **Accurate prediction:** existing cap 0.25.

Positive auxiliary caps sum to 2.0; not every task can attain every cap. Scores are deterministic on fixed task/action histories; auxiliary rewards are intentional training supervision. They have not been independently shown to optimize future model performance. Test-power still uses a finite known omission set, and repeated perceptual checks are not automatically useful experiments.

The starting and final protected counts are released only after submit in the public `repair_summary`. Training progress releases an aggregate reward, not per-case answers. Public evaluation reward breakdown omits unapplied auxiliary diagnostics.

## Compatibility

This is reward version `relative-start-v1`, environment `VisualRepairDelivery-v7`. Previous v6 model returns are historical, use the previous definition, and are not rescored. No old database is migrated in place. Absolute grader parity and finite exploit tests do not establish universal non-gameability.
