# User guide

## Manual task

1. Install and start the package using the README. Open http://127.0.0.1:8822/.
2. Choose a task, image condition and reward mode. Click **Start task**.
3. The approved reference is on the left and current candidate on the right. Expand **Public prompt, controls and language contract** to see exactly the information supplied to the agent, including action examples and supported Python syntax.
4. Enter one JSON experiment, for example `{"type":"interact","event":{"kind":"reset_scene"}}`. Each action consumes one budget unit. Use the displayed family-specific controls to choose useful probes yourself.
5. Edit the candidate source and **Save edited code**. **Run saved code** activates it. Compare images and choose tests that might expose an incomplete fix.
6. **Submit for grading** ends the episode. The GUI separately shows starting/final protected checks, improvement, terminal reward, costs and total return. A full grader pass does not mean the model verified it.
7. Step-history buttons replay observations in your current browser session. **Export trace** saves this public browser history. Persisted benchmark traces contain every model observation/call; they survive restarting the evaluator. Manual browser history is not reconstructed after a reload, but the current episode state resumes using its ID.

Episode IDs are kept in this browser's local storage. A pending manual step is saved before posting; retries reuse it. Keep its episode ID until the action is resolved. Tokens are never stored in local storage.

## Benchmark your locally served VLM

Serve an image-input VLM using an OpenAI-compatible `/v1/chat/completions` endpoint. The package does not install model weights, vLLM or GPU drivers; use your model server's supported installation. Any compatible local model is allowed; no current model ranking is implied.

Open **Benchmark a model**:

- Endpoint: e.g. `http://127.0.0.1:8000/v1`. In the Docker GUI use `http://host.docker.internal:8000/v1`; Linux compose supplies the host alias. The model server must accept connections from the evaluator container if using Docker.
- Model ID: exactly the name served by your server.
- Token: optional local-server bearer token, held only in memory.
- Scope: all 134 bundled tasks, one family, or the selected manual task.
- Attempts per task: one for a baseline, eight for task-level mixed-outcome groups.
- Concurrency: 1–8 independent episodes. Your model server must support the chosen load.
- Evidence condition follows the manual evidence selector. All attempts use evaluation rewards. The run configuration freezes the runtime/bank seal, task IDs, endpoint, model ID, evidence, repeats and concurrency.

Confirm local inference, then **Start benchmark**. Progress remains visible while the model runs. For a full bank × eight, 1,072 episodes are planned; this can take hours. Planned and interrupted episodes are not scores. A submitted failed repair or action-budget exhaustion is a completed outcome. Interrupted inference is unscored. Use **Download scores** for public JSON results, and retain the run directory for complete input/decision evidence.

After a restart, load the run ID and click **Resume run**. Re-enter a token if needed. Completed attempts stay completed. Pending saved decisions are reapplied idempotently. A malformed or truncated completed response stays blocked for that attempt rather than being replaced by another completion. Transport failures before a response was saved may be retried explicitly. The GUI does not automatically reconfigure providers or models.

The default GUI intentionally permits only local inference. No hosted key from earlier conversations is bundled, and no paid benchmark is launched by installation or tests.

### Native Codex adapter

Copy **only** `actor_client/` to a separate actor workspace and use an installed Codex CLI/sign-in. The evaluator code, banks and database must not be accessible to the actor process. From that actor-only directory:

```sh
python api_agent.py --endpoint http://127.0.0.1:8822 --task-id train-0000 \
  --output runs/my-model/train-0000 --model YOUR_AVAILABLE_CODEX_MODEL_ID --effort low
```

Use an available model ID from your own installation. The GUI is not a native Codex runner. It uses local OpenAI-compatible servers; the separate native adapter is an explicit single-episode workflow with saved calls. Merely installing this release does not execute Codex.

## Interpreting results

Full-repair fraction, absolute passed-check fraction and earned return measure different things. Report all three, plus action counts, interruptions, task family, budget, starting point and evidence arm. Eight samples of one task estimate a noisy task pass rate; they do not constitute eight independent task worlds. Compare models on matched tasks and use task-clustered intervals for repeated attempts. The GUI deliberately does not display a misleading independent-attempt confidence interval.

Mixed-group share uses only complete groups of eight on the same task/model/condition and counts both pass and fail before filtering. It is a proxy for outcome diversity, not measured gradients or training improvement. The bundled pool is public development data, not a private test set.

## Optional checkpoint priors

Initialize a v7 curriculum with the certified public training bank before recording actual model outcome groups:

```python
import json
from visual_repair_lab.v7.curriculum import Curriculum
c = Curriculum("var/curriculum.db", "checkpoint-1")
for record in json.load(open("banks/train.json"))["records"]:
    c.add(record)
```

Use `record-group` only with eight actual current-checkpoint outcome records containing distinct `attempt_id`, `status` and boolean `success`; interruptions do not count as failures. The GUI exports outcomes but does not silently launch calibration or write a training curriculum.

After a new checkpoint ID is created in the **same sealed v7 curriculum database**, import an older checkpoint's measured groups:

```sh
.venv/bin/visual-repair-lab inherit --database var/curriculum.db --policy-id checkpoint-2 \
  --source-policy checkpoint-1 --age 1 --decay 0.5
```

The prior uses Beta(1,1) plus discounted successes/failures. Its effective sample count is `8 × decay**age`; old observations remain tied to the source policy and group. Priors affect task selection only. They cannot enter current pass-rate reports, group filtering or the mixed-outcome headline. Selected tasks still require current-checkpoint measurements. Existing v6 curriculum files remain immutable and cannot be reopened under v7.

## Recovery and troubleshooting

- **Connection refused:** start the evaluator and verify the port. In Docker, localhost inside the container is not the host model server.
- **Model interruption:** inspect the saved public call or checkpoint; it is unscored. Check image support and JSON output. No raw credential-bearing exception is saved.
- **HTTP409:** another step won; resume using the episode ID. Same-episode parallel decisions are invalid.
- **Runtime/bank changed:** preserve the database and use a new path; do not edit metadata to bypass the source seal.
- **Timeout:** the prior state is retained. Retry the same saved action/request ID; the default worker timeout is 90 seconds via CLI.
- **Port already in use:** pass `--port 8824` to the CLI and open that port.
- **Nonzero return on an unchanged start:** terminal should be zero; total is normally negative from actions. Train mode can include separately earned experiment/comparison signals, which are not terminal payment for inherited correctness.
