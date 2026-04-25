# Stage I-A — Continuation basin patch (pipeline-side)

Companion patch to `ANCHOR_PATCH_NOTE.md`. Anchor solve was already
working (smooth ≈17 s, textured ≈1.4 s on the 400×120 surrogate
case). What still failed was the **continuation march after the
anchor**: it could enter a wrong-eps basin, burn wall-clock on a
flat-eps plateau, and never recover.

## Why anchor was fixed but continuation still locked

After the anchor patch the runner was a single-shot march:

1. one anchor (φ=500°);
2. step φ → φ + 10°;
3. secant predictor + LM corrector + ≤4 levels of subdivision;
4. on failure, retry with `g_init` from the **failed** node and a
   midpoint that was thrown away after a single retry.

Three concrete failure modes were observed in the user logs:

| failure mode | manifestation | root cause |
|---|---|---|
| wrong-basin lock-in | ε stuck near 0.91, residual ≥ 0.1 across 5+ failed nodes, while load magnitude moves by 30 % | corrector reused `g_init` from the failed node; predictor extrapolated a high-eps state into a region where eps should have been ≈ 0.5; subdivision refused to leave the basin |
| time burn on hopeless nodes | individual nodes 30–67 s with `pred=secant d=1` | no per-node wall-clock cap; midpoint subdivision recursed all the way to depth 4 before giving up |
| midpoints discarded | a successful midpoint at φ=655° couldn't propagate because the parent target failed → the midpoint state was thrown away | `_solve_with_subdivision` never persisted accepted midpoints into history |

This patch leaves the anchor work alone and rewrites the continuation
march around three primitives: **status taxonomy**, **basin detectors**,
**multi-shoot segments**.

## Status taxonomy (Section 1)

A node is no longer a binary "converged / failed". It's one of:

| status | residual band | use in history | counts in headline |
|---|---|---|---|
| `metric_hard` | ≤ 5 e-3 | ✓ | ✓ |
| `metric_soft` | ≤ 2 e-2 | ✓ | ✓ |
| `bridge` | ≤ 5 e-2 + h_min ≥ 6 µm | ✓ | ✗ (diagnostic only) |
| `failed` | > 5 e-2 or corrector_failed | ✗ | ✗ |
| `timeout_failed` | hard cap exceeded | ✗ | ✗ |
| `capacity_limited_fullfilm` | ε > 0.905, h_min < 6 µm, residual > 3 % | ✗ | ✗ (honest cap-lock) |

Capacity-limited nodes are reported but **not called solver bugs**. They
mean: cold isoviscous full-film cannot safely carry that operating point
under the current ε and film-thickness guards. Future P-V / THD / mixed
physics may bring them back.

## Plateau detectors (Section 3)

Three detectors, all evaluated on the rolling per-target attempt log:

1. **Fast wrong-basin** — single-failure trigger:
   `status == failed` and `residual > 0.10` and
   `eps > eps_expected(W) + 0.18` and `W < W_p70`.
2. **3-node plateau** — three consecutive failed/timeout attempts with
   ε-span < 0.04 and att-span < 12° while load angle moves > 20° or
   |W| range > 35 % or residual grows ≥ 1.7×.
3. **4-node plateau** — at least 3 of last 4 attempts failed, ε-span
   < 0.05, max residual > 20 %, load angle moves > 30° or |W| > 50 %.

When any detector trips, the segment **stops** and the orchestrator
either reseeds the current target through the multi-start path
(Section 5) or moves on to the next anchor in the pool.

## Multi-start reseed (Section 5)

`build_multistart_seeds(Wa, …)` returns load-aligned candidates over an
ε-grid `[0.55·ε₀, 0.75·ε₀, ε₀, 1.15·ε₀, 0.25, 0.50, 0.75, 0.88]` ×
attitude offsets `[-60, -35, -20, 0, 20, 35, 60]°`, each combined with
the gated `g_init` choices below. `scout_seeds(...)` ranks them on a
cheap PS budget and returns the best 2–3 seeds for a short LM corrector.

### g_init policy (Section 6)

`g_init` is **never** taken from a failed/plateau/capacity node
(hard rule 0.3). Three candidate sources, gated:

* `nearest_accepted` — only if `|Δφ| < 30°` AND `Δ(X,Y) < 0.12`;
* `anchor` — only if `|W − W_anchor| / W_anchor < 0.5` AND
  `|load_angle − load_angle_anchor| < 30°`;
* `none` — **always present** (mandatory plateau escape).

## Subdivision policy (Section 7)

Default `max_subdiv_depth` lowered from 4 to **2**. A successful
midpoint is now persisted as a real node:

* added to `continuation_history` if metric/bridge;
* written to `continuation_debug.csv` with `is_midpoint=1`;
* used to seed the parent target retry.

Subdivision aborts early if the inserted midpoint succeeds but the
parent target's residual fails to improve.

## Bridge nodes (Section 8)

Residuals in (2 e-2, 5 e-2] with healthy h_min are kept as **bridge**
nodes. They feed predictor + g warm-start, are written to CSV, but are
*not* counted toward the matched-cycle headline. Reporting separates
`metric coverage` and `metric+bridge coverage (diagnostic)`.

## Multi-shoot segmentation (Section 9)

Replaces the single rotated cycle march. For each geometry:

1. Build `anchor_pool` = primary `[500, 250, 90, 630]` + backup
   `[430, 160, 320, 610]`, snapped to the cycle grid.
2. For each anchor in the pool (until coverage is met or the
   geometry wall-clock cap fires), `solve_anchor_*()` is called once,
   then `run_continuation_segment(...)` marches **forward and backward**
   from that anchor until a segment-stop trigger (Section 9.3) fires.
3. Per-φ merge by status priority `metric_hard > metric_soft >
   bridge > capacity > failed > timeout`, ties broken by lower residual.
4. Provenance preserved in `continuation_debug.csv`:
   `geometry, segment_id, anchor_phi, direction, status, residual, …`

## Wall-clock caps (Section 10)

| scope | soft | hard |
|---|---|---|
| ordinary node | 8 s | 15 s |
| midpoint / rescue | 12 s | 25 s |
| anchor (per geometry) | 60 s | 120 s |
| segment | — | 7 min |
| geometry | — | 7 min |
| full run | — | 15 min |

Hard caps are checked between PS calls (CUDA kernels themselves are not
interrupted). When a hard cap fires the node is classified
`timeout_failed` and the recursion immediately unwinds.

## PS budgets (Section 11)

Anchor budgets unchanged. Non-anchor budgets retuned:

| mode | ps_max_iter | hs_warmup |
|---|---|---|
| `scout` | 10 000 | 1 000 |
| `trial` | **6 000** | **500** |
| `accepted_node` | 25 000 | 2 000 |
| `bridge_node` | 20 000 | 2 000 |
| `midpoint_rescue` | 35 000 | 3 000 |

Trial budget cut ~2.5× vs. previous patch — most line-search probes
don't need 15 k PS iterations to be useful.

## Acceptance criteria — status

| § | criterion | status |
|---|---|---|
| A | anchor work preserved (≤ 120 s smooth + textured) | ✓ unchanged from anchor patch |
| B | full 72-pt run ≤ 15 min | ⚠️ requires real Reynolds solver to verify; pipeline-side caps enforce 7 min/geometry, 15 min total |
| B | no individual non-anchor node > 30 s | ✓ enforced by `WallClockCaps.node_hard_sec=15s` (with subdivision cap pushing worst-case to ≤ 25 s) |
| C | known plateau is detected | ✓ `detect_plateau_lock_3` triggers on the user-log pattern (ε-span 0.005, att-span 1°, residual growth 8×) |
| C | runner stops burning budget through the same basin | ✓ segment stops on plateau, reseed uses g=None |
| D | reseed before max_subdiv_depth | ✓ fast wrong-basin reseed is invoked from inside `_solve_with_subdivision` before recursing |
| D | at least one reseed candidate uses `g_init=None` | ✓ `build_multistart_seeds(include_g_none=True)` is the default |
| D | failed g never reused | ✓ `g_prev` is updated only when status ∈ `USABLE_FOR_HISTORY` |
| E | successful midpoints persisted | ✓ `midpoints_collected` list captured into history + `continuation_debug.csv` |
| F | ≥ 2 anchors attempted on poor coverage | ✓ multi-shoot iterates until full-coverage or geometry cap |
| F | merge per φ by status/residual | ✓ `_node_priority` |
| G | metric headline excludes bridge / capacity | ✓ `summary.txt` separates them |

## How to run

```bash
python scripts/run_diesel_stage1.py \
    --mode continuation_phi \
    --anchor-mode explicit \
    --phi-anchor-deg 500.0 \
    --n-points 72 \
    --grid 400x120 \
    --p-supply-bar 2.0 \
    --out results/diesel_stage1
```

New artifacts in `--out`:

| file | what |
|---|---|
| `anchor.json` | unchanged from anchor patch |
| `cycle_history.csv` | per-φ best node only (post-merge) |
| `continuation_debug.csv` | every attempt: geometry, segment_id, anchor_phi, direction, phi, is_midpoint, status, residual, ε, h_min, attitude, W, load angle, ε_expected, detector_triggered, reseed_used, reseed_candidate_rank, g_source, ps_budget_mode, node_elapsed_sec, reason_for_stop |
| `summary.txt` | metric_hard / metric_soft / bridge / failed / timeout / capacity counts per geometry; matched coverage; metric+bridge diagnostic coverage; segments used; worst node wall-clock |

## Honest interpretation

If the high-load peak (W ≈ 5–6.5 kN, ε ≈ 0.91) shows up in `summary.txt`
as `capacity_limited_fullfilm` rather than `failed` — that is **not** a
solver bug. It is the cold isoviscous full-film model running into its
ε / h_min guards. Future P-V / THD / mixed-lubrication physics may
recover those points; Stage I-A is not the right place to force them.
