# Visual Repair Lab 7.0.0

A local, interactive visual-code repair environment. An agent sees an approved reference and a candidate, chooses experiments, edits Python, activates the edit, tests it and submits to a protected grader.

**v7 fixes inherited-correctness rewards and serialized history replay.** It adds a browser GUI for manual testing and benchmarks of a locally served VLM. Earlier distributions and model scores are not changed. This release does not train a model or claim that its rewards improve model weights.

## Install and open

Python 3.11–3.13, macOS or Linux. No API key or GPU is needed for the evaluator.

```sh
bash scripts/install.sh
bash scripts/start_lab.sh
```

Open **http://127.0.0.1:8822/**. The startup script uses the package directory regardless of the shell's current directory. It keeps episodes and run evidence in `var/`.

For a clean container:

```sh
docker compose up --build
```

The same GUI opens on port 8822. Episodes persist in the Docker volume. To run the standalone checks:

```sh
.venv/bin/python -m pip install -r requirements-test.lock
.venv/bin/python scripts/run_tests.py --output validation/local --timeout 420
.venv/bin/python scripts/validate_v7.py
```

## Use your model

Open **Benchmark a model**, enter your local OpenAI-compatible VLM endpoint and served model ID, select tasks and repetitions, confirm local inference, then start. The default is all 134 bundled development tasks × eight attempts. Set one repetition for a conventional baseline; eight estimates mixed-outcome task groups. For image conditions the model must support image-input chat messages.

The GUI restricts requests to loopback or Docker host aliases; it does not start remote hosted or paid inference. Authentication tokens stay in process memory. Provider interruptions are unscored. Saved model decisions and pending request IDs are reused after restart; malformed completed responses are preserved rather than regenerated. The benchmark saves public observations and calls under `var/episodes.runs/`.

A wiring-only smoke test is available, clearly labeled as scripted rather than a model baseline:

```sh
.venv/bin/python scripts/mock_model.py --port 8008
```

Use endpoint `http://127.0.0.1:8008/v1`, model `scripted-submit-smoke`. It always submits the starting code and earns no terminal reward. This tests your installation, not repair ability.

For native Codex sign-in, the separate `actor_client/api_agent.py` remains available; see [User guide](docs/USER_GUIDE_V7.md). It must run from an actor-only filesystem, not this trusted evaluator directory.

## What changed

- **Relative terminal reward:** unchanged correctness earns zero; improved full repair earns 10 before costs. Partial reward measures the fraction of initially failing checks repaired, with regression penalties. Absolute protected-check success stays separate.
- **Incremental state:** compressed, data-only checkpoints preserve current code, history, RNG, credit sets and ledgers. One worker executes one new action, with no reset or action-history replay.
- **Concurrent commits:** evaluation runs outside the SQLite write transaction. Commit checks the expected step; duplicate request IDs commit once. Runtime/bank seals prevent silent resume under changed code.
- **Training feedback:** progress uses schema-valid field agreement on the disjoint progress pool; no inherited field agreement is paid. Test-power credit is at most 0.1 per action, within the existing 0.5 episode cap. An exposure score does not prove the agent noticed a defect.
- **Checkpoint reuse:** optional discounted scheduling priors preserve old evidence without pretending it is a new checkpoint's measured pass rate.
- **Operator GUI:** manual controls, public input images/code, reward breakdown, final grades, local model benchmark progress, saved runs and JSON export.

Read the [user guide](docs/USER_GUIDE_V7.md), [reward specification](docs/REWARD_V7.md), [server design](docs/SERVER_V7.md), [delivery validation](DELIVERY_REPORT.md) and [limitations](docs/LIMITATIONS_V7.md).

Legacy modules are retained for historical replay. For new runs use the v7 console, `visual_repair_lab.make_env(spec=...)` or Gym ID `VisualRepairDelivery-v7`; direct imports of historical v5/v6 classes retain their original rewards.

## Trust boundary

Only public HTTP responses enter actor prompts. Never mount this evaluator package, banks, databases, private checkpoints, Docker socket or research reports into the actor environment. The browser is an operator tool, not an agent tool. The restricted Python interpreter and client separation are research controls, not a hardened arbitrary-code sandbox. Bundled tasks are public, development-used data; establish a separately generated and access-controlled test bank before reporting a private holdout result.
