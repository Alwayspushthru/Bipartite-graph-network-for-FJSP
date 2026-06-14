---
name: beam-vs-sampling-regime-split
description: Inference-time finding — deterministic beam beats sampling only on SD1, loses on SD2/BenchData
metadata:
  type: project
---

Full SD1/SD2/BenchData eval (model 10x5, width 8, 2026-06-03) of the new
deterministic beam search (`test_beam_strategy`, ranks by cumulative log-prob)
vs Sampling×8, in gap-% (beam − sampling, negative = beam better):

- SD1 30x10: −0.86 ✅, SD1 40x10: −1.61 ✅  (beam wins both)
- SD2 30x10+mix: +2.08 ❌, SD2 40x10+mix: +0.67 ❌
- BenchData Brandimarte +1.25, Hurink_edata +3.18, Hurink_rdata +2.31, Hurink_vdata +0.30 (beam loses all)

Clean split: beam wins only on SD1; loses on SD2 and every BenchData group.

**Diagnosis:** pure-probability beam does diversity-collapse (K beams share a
long high-prob prefix → "polish the mode"). Works when the policy mode is near
-optimal & distribution is sharp/low-flexibility (SD1); fails when the policy is
flat / OOD / high-flexibility (SD2 mix, real BenchData) where the mode is
unreliable and exploration matters → sampling's i.i.d. diversity wins.

**Next step:** stochastic beam search / Gumbel-top-k (sampling-without-
replacement) as the unifying fix — keeps beam structure + sampling diversity.
Secondary lever: critic-guided scoring `logP + λ·V(s')`, but must first validate
the value head's OOD calibration (it's trained only on 10x5). The unused critic
head is available at inference (`pi, _, h = ppo.policy(...)` in test.py).
