# 20-task toy comparison

**Toy-scale/offline adaptations, not original-paper benchmark reproductions.**
One grid environment; 20 fixed start/goal pairs; 30-action limit; 1,000 updates; three training seeds.
See [full protocol](../../docs/BENCHMARKS.md) for sources and differences.

| Method | Successes / 20 by seed | Mean ± sample SD | Parameters |
| --- | --- | --- | --- |
| Uni-LeWM + CEM | 7, 4 (incomplete) | 27.5% ± 10.6 pp | 58,224 |
| LeWM + CEM (official modules, toy scale) | 1, 6 (incomplete) | 17.5% ± 17.7 pp | 143,540 |
| Sub-LeWM + CEM (Sub-JEPA regularizer) | 3, 10 (incomplete) | 32.5% ± 24.7 pp | 143,540 |
| VLWM + CEM (independent paper adaptation) | 3, 5 (incomplete) | 20.0% ± 7.1 pp | 105,840 |
| FlexiWorld + CEM (discrete adaptation) | 8, 11 (incomplete) | 47.5% ± 10.6 pp | 202,688 |
| FlexiWorld Direct (same checkpoints) | 8, 8 (incomplete) | 40.0% ± 0.0 pp | 202,688 |
| Dreamer V1 (offline goal-conditioned) | 1 (incomplete) | 5.0% | 316,486 |
| Dreamer V2 (offline goal-conditioned) | 1 (incomplete) | 5.0% | 256,936 |
| Dreamer V3 (offline goal-conditioned) | 5 (incomplete) | 25.0% | 235,148 |
| Random | 5, 4, 3 | 20.0% ± 5.0 pp | 0 |
| Oracle shortest path (privileged) | 20 | 100.0% | 0 |

All learned models use the same 32 training episodes (384 original transitions), with eight held-out validation episodes.
The task list is reused across seeds: these are not 60 independent tasks. No test-score checkpoint selection.
Dreamer uses hindsight goal/reward supervision and the author actor-critic code; it is an offline adaptation.
Dreamer V1/V2 use six-channel images upscaled to 64×64; V3 uses current and goal 32×32 inputs.
Parameter counts describe the adapted networks; target/value copies and architecture differences prevent a parameter-matched algorithm ranking.
Recorded CPU wall times were measured during concurrent jobs and are not valid comparative speed measurements.
Raw task actions, training configurations, metrics, and source revisions are included for audit.

Missing runs: uni_lewm-seed2.json, lewm-seed2.json, sub_lewm-seed2.json, vlwm-seed2.json, flexiworld-seed2.json, flexiworld-direct-seed2.json, dreamerv1-seed1.json, dreamerv1-seed2.json, dreamerv2-seed1.json, dreamerv2-seed2.json, dreamerv3-seed1.json, dreamerv3-seed2.json
