# Actor-only package

This directory has no evaluator implementation, hidden bank, reference or grader. `client.py` uses only Python's standard library. Copy this directory into a separate actor filesystem; connect to the evaluator over HTTP. Do not mount the evaluator checkout or its database into the actor.

`api_agent.py` optionally uses an installed Codex CLI and existing sign-in. Run:

```
python api_agent.py --endpoint http://127.0.0.1:8810 --task-id drag-foundation-train --output ./attempt --model gpt-6.1-sol --effort low
```

Repeat the same command to resume. It reuses a saved pending decision/request ID. Do not delete the checkpoint after a timeout. Model calls, public prompts, observations and actions are preserved. It does not perform training or choose hidden tests.
