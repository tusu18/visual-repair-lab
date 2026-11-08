# Generator, hooks and acceptance

Each task contains family, hidden base world, ordered conditional rule list, action budget, starting point, split and scene seed. Train families are drag, linked brushing and contact dynamics. The four controlled base-world masks vary three additional factors; this is not nine independently sampled binary settings. One to three rules are selected from six primitives per family.

| Primitive | Drag hook | Chart hook | Physics hook |
|---|---|---|---|
| Limit | Card placement | Projection bounds | Velocity update |
| Gain | Pointer displacement | Projection scaling | Gravity update |
| Threshold | Resize transition | Projection threshold | Velocity dead band |
| Snap | Pointer release | Projected coordinates | Position update |
| Offset | Held-card drawing | Projection translation | Position update |
| Operation order | Zoom/snap displacement | Unzoom/snap/rezoom | Cap/gravity order |

Conditions use layout, view, zoom range, sign, velocity range or gravity sign. Each hook/operator/literal is whitelisted; arbitrary code, paths, imports and unknown fields are rejected.

Acceptance gates:

1. Disabling every added rule gives exact base source parity and tolerance-equivalent outputs over both finite banks.
2. The generated gold program executes in the actual bounded interpreter and has valid output shape on every protected case.
3. Removing each active rule changes at least two protected cases.
4. All reviewed rules preserve the generic task/control contract; no description promises a conflicting formula.
5. Duplicate observable behavior is rejected in the initial pool. Budget-edited descendants are explicit interventions and do not count as new mechanisms.
6. Neighboring floating-point values compare equal under the grader tolerance. Display rounding is never used to grade.
7. An ideal exact-footer reader identifies a passing source among finite candidate programs through a decision tree of public experiments, then edit/run/submit fit with two actions spare.
8. The final gate uses the SAME whole-family catalogue for every task, with no per-task hook/operation/condition hint. Observation aliases that change grading and excessive actual-branch experiment cost are rejected.

The release run proposed 200 tasks and accepted 134. The report in `banks/generation-report.json` retains every accepted/rejected specification and reason. The same family-wide decision tree is used for every target; budget admission charges its actual observation branch and edit/run/submit, while retaining the tree's worst-branch cost separately. The selected program is executed on the full accumulated public history and every protected check against the independently written reducer. Physics reset histories are checked for path invariance. Certificates are finite tested guarantees, not formal proofs for arbitrary source programs.

The ideal catalogue is a strong finite prior, not a universal program solver. Its decision tree and candidate indices stay evaluator-side. Exact numeric footers support the ideal reading abstraction; this is not an OCR accuracy result or a pure geometric-vision benchmark.

The generator can continue proposing and editing tasks, but its primitive vocabulary is finite. It cannot guarantee that any learner has a50% frontier. A learner unable to solve any valid task yields a parked/empty-boundary state, not fabricated mixed successes.
