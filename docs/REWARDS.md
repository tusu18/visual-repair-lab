# Rewards and interpretation

Train families retain v5's raw numeric comparison (absolute tolerance1e-6, relative1e-9), partial submission credit, progress on edit/run/submit, and test power only after changed code is activated. Progress and final banks have disjoint exact states but share mechanisms.

Every action costs0.025. Invalid actions cost another0.1. A full repair earns10; an incomplete submission earns `-2 + 8 * passed/total`. Unactivated submission costs0.25. Budget exhaustion without submission earns-2. A syntax-valid run never reports reference correctness before submission.

Train-mode positive caps total2.0: progress0.75, test power0.5, final-revision verification0.25, bound defect comparisons0.25, predictions0.25. Repeated code/tests cannot replenish caps. Comment-only changes, tolerance-equivalent edits and invalid outputs do not unlock test-power or final-revision verification credit. Regression loses0.75 times the fraction of previously passing progress cases lost. Incorrect bound claims/forecasts cost0.1. Reports use displayed precision; grading uses raw tolerances.

Added rules have omission mutants. Exposure, noticed mismatches, repair progress and final correctness remain separate. Exposure alone does not establish noticing. No required verification sequence or coverage quota is imposed.

The held-out detection family is evaluation-only: task rewards and terminal partial credit apply, but no private dense progress or mutant feedback is supplied. Private checking never updates the teacher.

The DAPO-style binary keep flag is separate from dense-reward variance. All-fail groups can contain different partial returns; all-pass groups can contain different action costs. They can still matter to an optimizer. Mixed-group share is a binary-outcome diversity proxy, not proof of gradients or learning.
