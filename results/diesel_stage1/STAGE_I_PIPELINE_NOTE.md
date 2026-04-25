# Stage I-A — Cold Quasi-Static Diesel Cycle

## Setup

- **Case**: Synthetic main bearing (D=110mm, L=55mm, c=70μm, 2000rpm, L/D=0.50)
- **Oil**: mineral at 105°C, η=0.010 Pa·s, iso-viscous (no PV, no thermal)
- **Load**: surrogate_heavyduty_v1 — Wiebe-like gas pulse + slider-crank inertia
- **Texture**: half-herringbone pump_to_edge, N=10, β=45°, d_g=15μm, taper=0.6
- **Feed**: point_unloaded, 2 bar, shell-fixed placement from φ_ref=90° ATDC
- **Grid**: 400×120 (coarse screening)
- **Solver**: stationary Payvar-Salant JFO + NR equilibrium with backtracking

**This is a synthetic case study. NOT a real engine geometry.**

## What can be stated

1. **Matched cycle comparison (headline)**:
   On the subset of crank angles where both smooth and textured equilibrium
   converged (31% of cycle at 36 points), textured geometry reduces
   cycle-mean friction power loss by **1.3%** (Pmean_smooth=419.5 W →
   Pmean_textured=413.9 W).

2. **Quasi-static validity**: squeeze-to-wedge ratio R_sq < 0.005 at all
   converged angles — quasi-static approximation is well-justified for
   this surrogate cycle.

3. **Textured convergence**: 27/36 angles converged (75%), vs 13/36 (36%)
   for smooth. Texture improves NR convergence, likely because the groove
   pumping creates a more structured pressure field.

4. **Smart seed**: load-jump-triggered seed reset prevents warm-start
   from trapping textured in high-ε spurious branches.

## What cannot be stated

1. **Full-cycle eq-vs-eq comparison is not available**: smooth equilibrium
   fails on ~64% of the cycle (high-load firing peak + low-load reversal
   zones). This is NOT a code bug — it is a limitation of stationary
   NR + periodic BC on thin films under rapid load changes.

2. **The 1.3% improvement does not represent the full cycle**: the matched
   subset covers mid-load angles (ε≈0.3-0.55) where both geometries are
   in a comfortable operating range. High-load and very-low-load regimes
   are excluded.

3. **Absolute Ploss values are not validated against engine data**: the
   surrogate load cycle is synthetic. Power loss magnitude should be
   treated as order-of-magnitude.

## Smooth no-equilibrium

Smooth bearing fails to find equilibrium on 64% of the surrogate cycle.
This is a continuation of Stage H findings (Report 11): smooth with
periodic BC at high load cannot establish stable NR convergence. This is
a physical boundary of the stationary model, not a solver defect.

Possible resolutions (not attempted in I-A):
- transient Reynolds with squeeze term
- groove BC (axial supply groove) — but this is physically inappropriate
  for main bearing without a continuous supply groove
- Dirichlet supply pressure on wider zone — but belt_wide destroys
  hydrodynamic wedge (Stage H finding)

## Cycle discretization

36-point run completed. 72-point run required for acceptance (п.5.6 ТЗ):
|Pmean_36 - Pmean_72| / Pmean_72 < 3%.

## Artifacts

- `cycle_history.csv` — per-angle raw data
- `compare_smooth_vs_textured.csv` — pairwise with matched flag
- `input_case.json` — full case definition
- `eps_vs_phi.png`, `hmin_vs_phi.png`, `ploss_vs_phi.png` — overlays
- `load_hodograph.png`, `wx_wy_vs_phi.png` — load cycle visualization
- `anchor.json` — anchor diagnostic (after I-A anchor patch, see below)

## Continuation basin fix

**Status:** see `CONTINUATION_BASIN_PATCH_NOTE.md`. Summary:

- The anchor patch unblocked the *entry* onto the branch but left the
  cycle march vulnerable to wrong-basin lock-in, flat-eps plateaus, and
  silent budget burn on hopeless nodes.
- The basin patch replaces the single-shot continuation with **basin-aware
  multi-shoot**:
  1. Node status taxonomy: `metric_hard / metric_soft / bridge / failed /
     timeout_failed / capacity_limited_fullfilm` — only metric statuses
     count toward the matched-cycle headline.
  2. Three plateau / wrong-basin detectors (fast 1-node, rolling 3-node,
     rolling 4-node) plus an explicit cap-lock detector.
  3. **Multi-start reseed** (load-aligned ε grid × attitude offsets,
     gated `g_init` choice with mandatory `g=None` candidate) before
     subdivision is allowed to recurse to its limit.
  4. **Midpoint persistence**: a successful midpoint is now a real node,
     added to history and to `continuation_debug.csv`.
  5. **Multi-shoot anchor pool**: primary `[500, 250, 90, 630]` +
     backup `[430, 160, 320, 610]`. Each anchor spawns a forward and
     a backward segment; per-φ merge by status priority.
  6. **Wall-clock caps** at node / midpoint / segment / geometry / full
     run scopes; hard cap fires `timeout_failed` and unwinds.
  7. **Honest reporting**: `summary.txt` separates metric, bridge, and
     capacity-limited counts. The high-load peak appears as
     `capacity_limited_fullfilm` rather than `failed` when it hits the
     ε / h_min guards — this is the physical cap of cold isoviscous
     full-film, not a solver defect.
- New CLI / artifacts: `continuation_debug.csv`, `summary.txt`. Existing
  `cycle_history.csv` now contains the per-φ best node after merge.
- Pipeline-side logic covered by `tests/test_basin_patch.py` (24 tests).

## Anchor policy fix

**Status:** see `ANCHOR_PATCH_NOTE.md` for the full pipeline-side patch
note. Summary:

- Default anchor was previously `argmin(|W|)` over the cycle and was
  solved through the generic angle-subdivision corrector. That path
  trapped the runner in an ill-conditioned region before the first
  accepted node (Stage I-A "stuck before first accepted point" symptom).
- Fix splits Stage I-A into two regimes:
  1. **Land on the branch** — `models/anchor_solver.py` with mild load
     homotopy `λ = [0.4, 0.6, 0.8, 1.0]` at a fixed, well-conditioned
     anchor angle (default φ_a = **500°** for `surrogate_heavyduty_v1`).
     Smooth anchor is accepted first; textured anchor is seeded from the
     smooth anchor state, with optional geometry continuation
     `α_tex = [0.33, 0.66, 1.0]` as rescue.
  2. **March the branch** — `models/continuation_runner.py::run_continuation_cycle`
     now takes an externally-solved `anchor_state` and skips its internal
     anchor solve. Existing predictor / corrector / subdivision /
     branch-jump logic operates only on angles after the anchor.
- PS budgets are now stage-dependent (anchor_stage_first / _later, trial,
  scout, accepted_node, midpoint_rescue). Trial evaluations are strictly
  cheaper than accepted evaluations; production PS budget is no longer
  spent on every candidate inside the corrector.
- New CLI flags on `scripts/run_diesel_stage1.py`:
  `--anchor-mode {explicit|from_legacy_matched_sector|scout_best}`,
  `--phi-anchor-deg`, `--legacy-history-csv`.
- Pipeline-side logic is covered by `tests/test_anchor_solver.py` (10/10).
