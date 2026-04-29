#!/usr/bin/env python
"""Stage J fu-2 Task 14.2 — failure-bucket post-processor.

Reads a run's ``data.npz`` / ``data_partial.npz`` and classifies
each step into a failure bucket. Produces
``stage_j_failure_diagnosis.txt`` in the run directory plus a
short summary on stdout.

Usage::

    python scripts/diagnose_stage_j_failures.py \\
        --run-dir results/diesel_thd_transient/<timestamp>... \\
        --config all

The classifier itself lives in
``models.diesel_coupling.failure_classifier``; this script is a
pure I/O wrapper. It consumes only the post-run npz arrays —
zero GPU, no transient solver invocation — so it is safe to run
on partial-result archives produced by an aborted config.

Buckets (see ``classify_failure``):
    ok / solver_budget / solver_residual / solver_nonfinite /
    picard_not_converged / picard_relax_floor / reject_at_anchor /
    mechanical_guard / physical_guard / unknown
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Local-import the classifier — adding the repo root to sys.path
# lets the script run from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.diesel_coupling.failure_classifier import (  # noqa: E402
    FAILURE_BUCKETS,
    BucketCounts,
    StepDiagnosticRow,
    aggregate_buckets,
    classify_failure,
)


# ─── Defaults ──────────────────────────────────────────────────────


# Firing sector defaults match the runner's default
# ``--peak-lo-deg`` / ``--peak-hi-deg`` (330° / 480° on the crank-
# angle axis modulo 720°). Override on the CLI for non-default runs.
DEFAULT_FIRING_LO_DEG = 330.0
DEFAULT_FIRING_HI_DEG = 480.0

# Conservative defaults for the solver thresholds the classifier
# needs. Real values come from ``ausas_options`` saved alongside
# the data; these only kick in if the npz pre-dates that save.
DEFAULT_AUSAS_TOL = 1e-6
DEFAULT_AUSAS_MAX_INNER = 5000
DEFAULT_MAX_MECH_INNER = 8
DEFAULT_MECH_RELAX_MIN = 0.03125


# ─── npz loading ───────────────────────────────────────────────────


def _find_data_npz(run_dir: str) -> str:
    """Prefer ``data.npz`` (full run); fall back to ``data_partial.npz``
    (aborted run)."""
    for fname in ("data.npz", "data_partial.npz"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No data.npz / data_partial.npz in {run_dir!r}")


def _resolve_ausas_thresholds(
    npz: Dict[str, Any],
) -> Tuple[float, int]:
    """Pull ``tol`` / ``max_inner`` from the saved ``ausas_options``
    object array (length-1 list of dict). If absent, use the
    DEFAULT_*."""
    raw = npz.get("ausas_options")
    if raw is None:
        return DEFAULT_AUSAS_TOL, DEFAULT_AUSAS_MAX_INNER
    arr = np.asarray(raw)
    if arr.size == 0:
        return DEFAULT_AUSAS_TOL, DEFAULT_AUSAS_MAX_INNER
    opts = arr.flat[0] or {}
    tol = float(opts.get("tol", DEFAULT_AUSAS_TOL))
    max_inner = int(opts.get("max_inner", DEFAULT_AUSAS_MAX_INNER))
    return tol, max_inner


def _resolve_mech_relax_min(
    npz: Dict[str, Any],
) -> float:
    """The runner doesn't currently save ``mech_relax_min`` as a
    scalar; infer it from ``stage_j_mech_relax_min_seen`` arrays
    (the lower envelope across all runs is a tight bound). Falls
    back to ``DEFAULT_MECH_RELAX_MIN``."""
    arr = npz.get("stage_j_mech_relax_min_seen")
    if arr is None:
        return DEFAULT_MECH_RELAX_MIN
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return DEFAULT_MECH_RELAX_MIN
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return DEFAULT_MECH_RELAX_MIN
    floor = float(np.min(finite))
    # Defensive lower bound — we only trust the inferred value if
    # it's plausibly below the legacy 0.03125 default; otherwise
    # the lower envelope is just the un-shrunk relax_initial.
    if floor < DEFAULT_MECH_RELAX_MIN:
        return floor
    return DEFAULT_MECH_RELAX_MIN


def _resolve_max_mech_inner(
    npz: Dict[str, Any],
) -> int:
    """Same trick as ``_resolve_mech_relax_min`` but for the upper
    envelope of ``stage_j_n_trials``."""
    arr = npz.get("stage_j_n_trials")
    if arr is None:
        return DEFAULT_MAX_MECH_INNER
    a = np.asarray(arr, dtype=int)
    if a.size == 0:
        return DEFAULT_MAX_MECH_INNER
    cap = int(np.max(a))
    if cap > DEFAULT_MAX_MECH_INNER:
        return cap
    return DEFAULT_MAX_MECH_INNER


# ─── Per-config classification ─────────────────────────────────────


def _per_step_rows(
    npz: Dict[str, Any],
    ic: int,
    step_lo: int, step_hi: int,
    *,
    ausas_tol: float, ausas_max_inner: int,
    max_mech_inner: int, mech_relax_min: float,
) -> List[StepDiagnosticRow]:
    """Build ``StepDiagnosticRow`` instances for steps
    ``[step_lo, step_hi)`` of config ``ic``."""
    rejection_reasons = np.asarray(npz["stage_j_rejection_reason"])
    fp_conv = np.asarray(npz["stage_j_fp_converged"])
    n_trials = np.asarray(npz["stage_j_n_trials"])
    mech_relax_min_seen = np.asarray(npz["stage_j_mech_relax_min_seen"])
    ausas_n_inner = np.asarray(npz["ausas_n_inner"])
    ausas_residual = np.asarray(npz["ausas_residual"])
    ausas_converged = np.asarray(npz["ausas_converged"])
    pmax = np.asarray(npz["pmax"])
    theta_max = np.asarray(
        npz.get("ausas_theta_max", np.ones_like(pmax)))
    solver_success = np.asarray(npz["solver_success"])

    rows: List[StepDiagnosticRow] = []
    for step in range(step_lo, step_hi):
        rr_raw = rejection_reasons[ic, step]
        rr = "" if rr_raw is None else str(rr_raw)
        rows.append(StepDiagnosticRow(
            is_failure=not bool(solver_success[ic, step]),
            rejection_reason=rr,
            ausas_n_inner=int(ausas_n_inner[ic, step]),
            ausas_residual=float(ausas_residual[ic, step]),
            ausas_converged=bool(ausas_converged[ic, step]),
            p_max=float(pmax[ic, step]),
            theta_max=float(theta_max[ic, step]),
            fp_converged=bool(fp_conv[ic, step]),
            n_trials=int(n_trials[ic, step]),
            mech_relax_min_seen=float(mech_relax_min_seen[ic, step]),
            ausas_tol=ausas_tol,
            ausas_max_inner=ausas_max_inner,
            max_mech_inner=max_mech_inner,
            mech_relax_min=mech_relax_min,
        ))
    return rows


def _step_range_for_config(
    npz: Dict[str, Any], ic: int,
) -> Tuple[int, int]:
    """For aborted configs, only the first ``steps_completed[ic]``
    steps are real; everything beyond is unwritten zeros (which
    would otherwise show up as ``solver_success=False`` and
    inflate every bucket). Returns ``[0, n_real)``."""
    steps_completed = npz.get("steps_completed")
    if steps_completed is None:
        # Fall back to full array length if missing.
        return 0, int(np.asarray(npz["pmax"]).shape[1])
    n_real = int(np.asarray(steps_completed)[ic])
    return 0, n_real


def _firing_mask(
    phi_crank_deg: np.ndarray,
    step_lo: int, step_hi: int,
    *,
    firing_lo_deg: float,
    firing_hi_deg: float,
) -> np.ndarray:
    """Boolean mask (length = step_hi - step_lo) for steps whose
    phi modulo 720° falls in the firing sector."""
    phi = np.asarray(phi_crank_deg)[step_lo:step_hi]
    phi_mod = phi % 720.0
    return (phi_mod >= firing_lo_deg) & (phi_mod <= firing_hi_deg)


def _last_cycle_range(
    npz: Dict[str, Any],
    step_lo: int, step_hi: int,
) -> Tuple[int, int]:
    """Restrict to the last cycle: max(``last_start``, step_lo)
    through step_hi."""
    last_start = int(npz.get("last_start", 0))
    return max(last_start, step_lo), step_hi


# ─── Reporting ─────────────────────────────────────────────────────


def _format_breakdown(
    title: str,
    counts_full: BucketCounts,
    counts_last: BucketCounts,
    counts_firing: BucketCounts,
) -> List[str]:
    """One block of the ``Failure breakdown`` section — three
    columns: full / last_cycle / firing_sector."""
    lines = [title]
    lines.append("  bucket                        full    last    firing")
    for bucket in FAILURE_BUCKETS:
        if bucket == "ok":
            continue
        nf = counts_full.per_bucket.get(bucket, 0)
        nl = counts_last.per_bucket.get(bucket, 0)
        nfir = counts_firing.per_bucket.get(bucket, 0)
        if nf == 0 and nl == 0 and nfir == 0:
            continue
        lines.append(
            f"  {bucket:<28} {nf:>5d}   {nl:>5d}   {nfir:>5d}")
    lines.append(
        f"  {'-- total --':<28} "
        f"{counts_full.n_total:>5d}   "
        f"{counts_last.n_total:>5d}   "
        f"{counts_firing.n_total:>5d}")
    lines.append(
        f"  {'-- failures --':<28} "
        f"{counts_full.n_failures:>5d}   "
        f"{counts_last.n_failures:>5d}   "
        f"{counts_firing.n_failures:>5d}")
    dom = counts_full.dominant_failure_bucket()
    if dom is not None:
        n_dom = counts_full.per_bucket.get(dom, 0)
        frac = counts_full.frac_of_failures(dom)
        lines.append(
            f"  dominant (full)              : {dom} "
            f"({n_dom}/{counts_full.n_failures} = {frac:.0%})")
    return lines


def _format_first_failing(
    rows: List[StepDiagnosticRow],
    *,
    phi_crank_deg: np.ndarray,
    step_lo: int,
    n_max: int,
) -> List[str]:
    lines = [f"First {n_max} failing steps:"]
    lines.append(
        "  step   phi      bucket               n_inner  residual    "
        "fp_converged  relax_min  reason")
    n_emitted = 0
    for k, r in enumerate(rows):
        if not r.is_failure:
            continue
        if n_emitted >= n_max:
            break
        bucket = classify_failure(r)
        phi = float(phi_crank_deg[step_lo + k])
        residual_s = (f"{r.ausas_residual:9.3e}"
                      if np.isfinite(r.ausas_residual) else "  nan    ")
        relax_s = (f"{r.mech_relax_min_seen:8.5f}"
                   if np.isfinite(r.mech_relax_min_seen) else "    nan ")
        rr_short = (r.rejection_reason[:60]
                    if len(r.rejection_reason) <= 60
                    else r.rejection_reason[:57] + "...")
        lines.append(
            f"  {step_lo + k:>4d}  {phi:>6.1f}°  "
            f"{bucket:<20} {r.ausas_n_inner:>6d}  {residual_s}  "
            f"{str(r.fp_converged):>5}        "
            f"{relax_s}  {rr_short}")
        n_emitted += 1
    if n_emitted == 0:
        lines.append("  (no failing steps in this slice)")
    return lines


def diagnose_run(
    run_dir: str,
    *,
    config_filter: str = "all",
    firing_lo_deg: float = DEFAULT_FIRING_LO_DEG,
    firing_hi_deg: float = DEFAULT_FIRING_HI_DEG,
    first_n_failing: int = 10,
) -> str:
    """Top-level entry — produces the full report string. Pure
    function (no side effects on disk); the CLI wrapper writes it."""
    data_path = _find_data_npz(run_dir)
    npz = dict(np.load(data_path, allow_pickle=True))
    labels = [str(s) for s in np.asarray(npz["labels"]).tolist()]
    phi_crank_deg = np.asarray(npz["phi_crank_deg"], dtype=float)
    aborted = np.asarray(npz.get(
        "aborted", np.zeros(len(labels), dtype=bool)))
    abort_reason = np.asarray(npz.get(
        "abort_reason", np.array([""] * len(labels), dtype=object)))
    steps_completed = np.asarray(npz.get(
        "steps_completed", np.full(len(labels),
                                     phi_crank_deg.shape[0])))

    ausas_tol, ausas_max_inner = _resolve_ausas_thresholds(npz)
    mech_relax_min = _resolve_mech_relax_min(npz)
    max_mech_inner = _resolve_max_mech_inner(npz)

    if config_filter == "all":
        target_indices = list(range(len(labels)))
    else:
        if config_filter not in labels:
            raise SystemExit(
                f"Config {config_filter!r} not found in run "
                f"{run_dir!r}; available: {labels}")
        target_indices = [labels.index(config_filter)]

    out: List[str] = []
    out.append(f"Run: {os.path.basename(os.path.normpath(run_dir))}")
    out.append(f"Source: {os.path.basename(data_path)}")
    out.append("")
    out.append("Solver thresholds (used by classifier):")
    out.append(f"  ausas_tol           : {ausas_tol:.3e}")
    out.append(f"  ausas_max_inner     : {ausas_max_inner}")
    out.append(f"  max_mech_inner      : {max_mech_inner}")
    out.append(f"  mech_relax_min      : {mech_relax_min:.5f}")
    out.append(f"  firing_sector_deg   : "
                 f"[{firing_lo_deg:.1f}, {firing_hi_deg:.1f}]")
    out.append("")

    for ic in target_indices:
        cfg_label = labels[ic]
        out.append("=" * 64)
        out.append(f"Config: {cfg_label}")
        if bool(aborted[ic]):
            out.append(f"  status        : ABORTED")
            out.append(
                f"  abort_reason  : {abort_reason[ic]}")
        out.append(f"  steps_completed : {int(steps_completed[ic])}")
        out.append("")

        step_lo, step_hi = _step_range_for_config(npz, ic)
        if step_hi <= step_lo:
            out.append("  (no completed steps to classify)")
            out.append("")
            continue

        rows_full = _per_step_rows(
            npz, ic, step_lo, step_hi,
            ausas_tol=ausas_tol, ausas_max_inner=ausas_max_inner,
            max_mech_inner=max_mech_inner,
            mech_relax_min=mech_relax_min,
        )
        last_start, last_end = _last_cycle_range(npz, step_lo, step_hi)
        if last_end > last_start:
            rows_last = rows_full[last_start - step_lo: last_end - step_lo]
        else:
            rows_last = []
        firing_mask_full = _firing_mask(
            phi_crank_deg, step_lo, step_hi,
            firing_lo_deg=firing_lo_deg, firing_hi_deg=firing_hi_deg,
        )
        rows_firing = [r for r, m in zip(rows_full, firing_mask_full)
                        if m]

        counts_full = aggregate_buckets(rows_full)
        counts_last = aggregate_buckets(rows_last)
        counts_firing = aggregate_buckets(rows_firing)

        out.extend(_format_breakdown(
            f"Failure breakdown — {cfg_label}",
            counts_full, counts_last, counts_firing,
        ))
        out.append("")
        out.extend(_format_first_failing(
            rows_full,
            phi_crank_deg=phi_crank_deg,
            step_lo=step_lo,
            n_max=first_n_failing,
        ))
        out.append("")

    return "\n".join(out) + "\n"


# ─── CLI ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage J fu-2 — failure-bucket post-processor.")
    p.add_argument("--run-dir", required=True,
                    help="Path to a transient-run directory (must "
                         "contain data.npz or data_partial.npz).")
    p.add_argument("--config", default="all",
                    help="Config label to diagnose, or 'all' "
                         "(default).")
    p.add_argument("--firing-lo-deg", type=float,
                    default=DEFAULT_FIRING_LO_DEG,
                    help=("Firing-sector lower bound on crank angle "
                          f"(default {DEFAULT_FIRING_LO_DEG})."))
    p.add_argument("--firing-hi-deg", type=float,
                    default=DEFAULT_FIRING_HI_DEG,
                    help=("Firing-sector upper bound on crank angle "
                          f"(default {DEFAULT_FIRING_HI_DEG})."))
    p.add_argument("--first-n-failing", type=int, default=10,
                    help="How many failing steps to dump in detail "
                         "(default 10).")
    p.add_argument("--output", default=None,
                    help="Output path for the report (default: "
                         "<run-dir>/stage_j_failure_diagnosis.txt).")
    p.add_argument("--no-write", action="store_true", default=False,
                    help="Skip writing the report file; print only.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = diagnose_run(
        args.run_dir,
        config_filter=args.config,
        firing_lo_deg=args.firing_lo_deg,
        firing_hi_deg=args.firing_hi_deg,
        first_n_failing=args.first_n_failing,
    )
    print(report)
    if not args.no_write:
        out_path = (args.output if args.output
                     else os.path.join(args.run_dir,
                                          "stage_j_failure_diagnosis.txt"))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"[diagnose] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
