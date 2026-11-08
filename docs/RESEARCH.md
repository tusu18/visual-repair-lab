# Research mapping and limits

- [PLR](https://proceedings.mlr.press/v139/jiang21b.html) motivates replay according to estimated learning potential. Our implementation uses empirical binary variance; it does not reproduce PLR's TD-error score.
- [ACCEL](https://arxiv.org/abs/2203.01302) motivates editing environments around a learner's ability. We add/drop typed rules or change budget, then rerun acceptance checks. We do not claim ACCEL's training results.
- [PAIRED](https://arxiv.org/abs/2012.02096) motivates distinguishing difficult tasks from unsolvable ones. Here an explicit finite-catalogue ideal policy supplies a solvability upper bound. There is no learned antagonist, adversary or minimax-regret training.
- [Learning to Reason at the Frontier of Learnability](https://arxiv.org/abs/2502.12272) motivates sampling by `p*(1-p)`. We estimate p from eight completed episodes per task/checkpoint. The ideal maximum is p=0.5, but finite sampling, a limited pool and imperfect editing do not guarantee that distribution.
- [DAPO](https://arxiv.org/abs/2503.14476) motivates filtering uniform-outcome groups. Our retained groups are mixed by construction. We report unfiltered mixed share and generation cost; no optimizer, token-level loss or training gain is implemented or claimed.

The work adapts these ideas to visual software repair. It is not a literature originality claim or a replication of the complete algorithms.

Fixed and adaptive proxy arms use the same frozen train-program corpus and numeric observations. A program catalogue is a privileged representational prior; it is not an oracle lookup of the current target, but it is much stronger than an untrained general coding agent. The policy chooses experiments and writes the selected candidate based on public evidence; success is never assigned by a coin flip. It has no private evaluation programs.

Three stages are distinct: train sampling diagnostics, native model calibration, and frozen private evaluation. Only the last supplies model evaluation scores. Private outcomes do not update the teacher. A frozen model can demonstrate sampling effects without demonstrating learning.

The historical48-action four-model study did not contain eight repeated attempts per task/profile. Its12/12 outcomes cannot establish a measured0% mixed-group rate. We report a directly measured fixed-pool control rather than manufacture that number.
