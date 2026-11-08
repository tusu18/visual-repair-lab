# Known limits and delivery scope

This is a validated local research delivery, not certification for unrestricted frontier-lab deployment.

- No VLM weights were trained. There is no before/after learning experiment, universal grader-security proof, or new v7 model ranking.
- The 134-task bundle is public and development-used. v7 reuses its task content to isolate software/reward fixes; it is not a fresh holdout. The earlier v6 baseline is preserved under its original rewards and protocol qualification.
- Three generator families and finite rule primitives remain. Private detection overlay is supported by the evaluator, but no private test bank is shipped or inferred from the operator GUI.
- Finite independent-reference parity and adversarial regression tests cover known cases, not all possible programs. Aggregate field-progress can be optimized without solving whole protected cases. Its alignment with eventual training success remains unvalidated.
- Test-power witnesses are finite omission mutants; exposure is not proof of noticing or a good explanation. Per-action credit is capped, but this is not a complete experiment-quality metric.
- A task whose initial code is already fully correct cannot earn terminal repair reward under the user's no-unchanged-payment rule. Filter or separately sample correct-candidate audits for training.
- Process startup, rendering and checkpoint serialization still cost CPU time. Very long candidate event histories still cost candidate computation. No distributed multi-host evaluator or arbitrary-code OS sandbox is shipped.
- GUI models must expose a local OpenAI-compatible image-input chat endpoint and JSON-object output. Remote paid inference is disabled. The native Codex adapter is separate; multi-provider orchestration and GPU/model-server installation are not bundled.
- The benchmark records task-level repeated outcomes without claiming independent repeated worlds. User-operated results need adequate distinct tasks, matched comparisons and clustered statistical analysis before research conclusions.
- For model training use the HTTP actor boundary and separate mounts/network controls. A trusted in-process Gym environment can inspect private attributes and is not an actor sandbox.
