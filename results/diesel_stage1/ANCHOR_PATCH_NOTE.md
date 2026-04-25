# Stage I-A — Anchor policy fix (pipeline-side patch)

## Why min-load anchor was wrong

The previous Stage I-A continuation runner picked the global *minimum-load*
crank angle as the anchor (`run_diesel_stage1.py::_run_continuation`,
`idx_min_load = int(np.argmin(loads))`) and solved it via the generic
angle-subdivision corrector. Three things failed in this combination:

1. **Min-load anchor is not well-conditioned.** At very low |W| the
   journal eccentricity equilibrium is shallow and ill-conditioned: small
   changes in (X, Y) produce small changes in F, so the Jacobian is poorly
   scaled and Newton corrections are dominated by relative noise from PS
   residual rather than load equilibrium. The runner spends production-grade
   PS budget per trial in a region where there isn't a strong attractor.

2. **The generic angle-subdivision path is the wrong tool.** Subdivision
   bisects in φ between two *already-accepted* states. At step 0 there is
   no history to bisect against, so subdivision degrades to repeated
   reseeding around the same ill-conditioned angle.

3. **Continuation-in-φ cannot start from min load.** Even when the anchor
   eventually solves, the next angle along φ is in a region where load is
   ramping up rapidly, so the secant predictor has to extrapolate into a
   regime the previous accepted state never saw.

## What replaced it

Anchor solve is now a **separate entry procedure** to land on the branch.
Continuation in φ runs only afterwards. New layout:

```
models/anchor_solver.py
  pick_anchor_phi(...)       -- selection policy
  solve_anchor_smooth(...)   -- mild load homotopy at fixed φ_a
  solve_anchor_textured(...) -- smooth-seeded + optional geom continuation
```

`models/continuation_runner.py::run_continuation_cycle(...)` now accepts
an externally-solved `anchor_state` and skips its internal anchor solve.
The legacy fallback path is preserved for non-Stage I-A callers but is
not used by Stage I-A.

### 1. Anchor selection (Section 1)

`pick_anchor_phi(phi_targets, load_fn, mode=...)` supports three modes,
with the same hard rule across all of them: never pick global min-load or
global max-load.

| mode                          | source of φ_a                              |
|-------------------------------|--------------------------------------------|
| `explicit`                    | `--phi-anchor-deg` (default **500.0°**)    |
| `from_legacy_matched_sector`  | midpoint of longest contiguous sector where both smooth+textured were accepted in a prior run; falls back to **500.0°** if legacy info is absent |
| `scout_best`                  | medium-load (~25–60 percentile) candidate with lowest control residual under a cheap scout PS budget |

For the current `surrogate_heavyduty_v1` 72-point fallback case the
pragmatic default is `phi_anchor_deg=500.0`, |W|≈2.3 kN, which sits in the
post-firing recovery sector — well-conditioned, with moderate load and no
peak-firing nonlinearity.

### 2. Smooth anchor: load homotopy (Section 3)

At fixed φ_a we solve `F_h(X, Y) = λ·W_a` for a short λ-schedule and warm-
start (X, Y, g_init) between stages.

- Default schedule:  `λ = [0.4, 0.6, 0.8, 1.0]`  (does **not** start at 0)
- Fallback schedule: `λ = [0.5, 0.75, 1.0]`
- Local corrector: damped Newton with LM safeguard, **hard-cap 8 NL iters**
  per λ-stage, target 4–6 (Section 6.1).
- Same shell placement, same geometry, same solver path; only |W| ramps.

### 3. Textured anchor: smooth-seeded (Section 4)

1. **First attempt:** direct textured solve at α_tex=1.0 from the accepted
   smooth anchor `(X, Y, g)`.
2. **Rescue:** short geometry continuation `α_tex = [0.33, 0.66, 1.0]`,
   warm-starting `(X, Y, g)` between α-stages. All other parameters fixed.
3. **No scratch solve at min-load** for textured.

### 4. Adaptive PS budgets (Section 5)

Stage-dependent budgets via mode-aware `eval_factory(mode_str)`. Trial
evaluations (Jacobian probes, line-search trial points) are **strictly
cheaper** than accepted evaluations, so we don't burn production-grade PS
budget on every candidate inside the corrector.

| mode                  | ps_max_iter | hs_warmup_iter |
|-----------------------|-------------|----------------|
| `scout`               | 12 000      |  1 000         |
| `trial`               | 15 000      |  2 000         |
| `anchor_stage_first`  | 100 000     | 25 000         |
| `anchor_stage_later`  |  80 000     |  5 000         |
| `accepted_node`       |  25 000     |  2 000         |
| `midpoint_rescue`     |  50 000     | 10 000         |

Anchor first stage gets a heavy hs warmup (cold pressure field). Later
λ-stages reuse the warm `g_init` and use a much lighter warmup.

## Acceptance criteria — status

| Criterion                                                              | Status |
|------------------------------------------------------------------------|--------|
| A. runner no longer picks global min-load anchor by default            | ✓ enforced by `pick_anchor_phi` (`test_pick_anchor_never_picks_global_min_or_max`) |
| A. anchor solve does not use generic angle subdivision                 | ✓ `solve_anchor_smooth` has no `load_fn` param, no φ-recursion (`test_smooth_anchor_does_not_use_subdivision`) |
| A. default anchor lands in medium-load stable sector                   | ✓ `phi_anchor=500°`, |W|≈2.3 kN |
| C. textured from smooth anchor; geometry rescue if direct fails        | ✓ `solve_anchor_textured` (`test_textured_anchor_direct_from_smooth`, `test_textured_anchor_geometry_rescue`) |
| D. continuation runner refuses to start without accepted anchor        | ✓ `run_continuation_cycle(anchor_state=...)` is the Stage I-A path |
| E. existing continuation logic preserved                               | ✓ `_solve_with_subdivision` unchanged in essence; only PS budget threading is added |

Wall-clock criteria (B: smooth anchor accepted within ~60 s on 400×120;
upper bound 120 s) require the Reynolds solver (`reynolds_solver`) which
is not installed in the patch CI sandbox. The pipeline-side logic is fully
exercised by `tests/test_anchor_solver.py` (10/10 green) using a synthetic
F(X,Y) of comparable nonlinearity.

## How to run on the current fallback case

```bash
python3 scripts/run_diesel_stage1.py \
    --mode continuation_phi \
    --anchor-mode explicit \
    --phi-anchor-deg 500.0 \
    --n-points 72 \
    --grid 400x120 \
    --out results/diesel_stage1
```

`--anchor-mode scout_best` and `--anchor-mode from_legacy_matched_sector`
(plus `--legacy-history-csv path/to/cycle_history.csv`) are also wired up.

The runner writes `results/diesel_stage1/anchor.json` with the full anchor
diagnostic (Section 9): chosen mode, φ_a, candidate list, selection
reason, λ-stage log (X, Y, ε, residual, NR iters, wall-time), textured
direct-vs-rescue path, PS budget table, and final anchor states.
