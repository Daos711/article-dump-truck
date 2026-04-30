"""Stage J fu-2 Task 29 — dump writer for failed Ausas one-step calls.

Saves the real input state of a failed ``ausas_unsteady_one_step_gpu``
call as a ``.npz`` so the gpu-reynolds-side
``replay_ausas_one_step_dump.py`` can reproduce the failure
standalone (no diesel runner, no Verlet/Picard) and locate the
first NaN field/index/iteration.

The diesel runner does NOT decode the dump — the contract here is
write-only. Mandatory keys are pinned by
``MANDATORY_DUMP_KEYS`` and validated by
``validate_dump_npz`` so the solver-side replay can rely on the
schema.

Pure I/O. No GPU. No Verlet/Picard math.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Mandatory keys ────────────────────────────────────────────────


MANDATORY_DUMP_KEYS: Tuple[str, ...] = (
    # Solver inputs
    "H_prev",
    "H_curr",
    "P_prev",
    "theta_prev",
    "dt_s",
    "d_phi",
    "d_Z",
    "periodic_phi",
    "periodic_z",
    "bc_z_low",
    "bc_z_high",
    "bc_phi_low",
    "bc_phi_high",
    "solver_kwargs",
    # Step metadata
    "step",
    "substep",
    "trial",
    "commit",
    "phi_deg",
    "eps_x",
    "eps_y",
    "config_label",
    "trial_kind",
    "texture_kind",
    "groove_preset",
    "cavitation",
    "n_phi",
    "n_z",
    # Solver result + nonfinite diagnostics
    "converged",
    "failure_kind",
    "first_nan_field",
    "first_nan_index",
    "first_nan_is_ghost",
    "first_nan_is_axial_boundary",
    "first_nan_is_phi_seam",
    "nan_iter",
    "residual",
    "residual_linf",
    "residual_rms",
    "residual_l2_abs",
    "n_inner",
    "nonfinite_count",
    # Shape diagnostics
    "P_shape_raw",
    "theta_shape_raw",
    "expected_physical_shape",
    "expected_padded_shape",
    "P_is_padded",
    "theta_is_padded",
    # Commit-semantics metadata (Task 32 sync)
    "final_trial_status",
    "committed_state_status",
    "accepted_state_source",
    "committed_state_is_finite",
)


OPTIONAL_FORCE_INPUT_KEYS: Tuple[str, ...] = (
    "P_raw",
    "theta_raw",
    "P_physical",
    "theta_physical",
    "phi_grid",
    "z_grid",
    "integration_weights_phi",
    "integration_weights_z",
    "pscale",
    "eta_eff",
    "T_eff",
    "H_min",
    "H_max",
    "F_hyd_x",
    "F_hyd_y",
    "pmax_nd",
    "pmax_dim",
    "h_min",
)


# ─── Dump configuration + counters ─────────────────────────────────


@dataclass
class DumpConfig:
    """Runtime config for the dump path.

    ``directory`` None disables the dump path entirely (production
    default). When set, the runner writes up to ``limit`` ``.npz``
    files per run, then increments ``suppressed`` instead.
    """
    directory: Optional[str] = None
    limit: int = 20
    include_force_inputs: bool = True
    # Trigger toggles — operator can disable individual triggers
    # if a known-failure pattern dominates and would otherwise burn
    # the limit on rows the agent already understands. Defaults
    # match the spec.
    on_nonfinite_state: bool = True
    on_invalid_input: bool = True
    on_residual_nan: bool = True
    on_converged_false: bool = True
    on_n_inner_at_max: bool = True
    on_force_nan: bool = True


@dataclass
class DumpCounters:
    """Per-run counters surfaced in summary."""
    written: int = 0
    suppressed_after_limit: int = 0
    write_failed: int = 0
    by_trigger: Dict[str, int] = field(default_factory=dict)


# ─── Trigger evaluation ────────────────────────────────────────────


def evaluate_triggers(
    *,
    cfg: DumpConfig,
    converged: bool,
    failure_kind: str,
    residual: float,
    n_inner: int,
    n_inner_max: int,
    F_hyd_x: float,
    F_hyd_y: float,
) -> List[str]:
    """Return the list of trigger tags that fire for this call.

    An empty list means "no dump needed". Used by the adapter to
    decide whether to write at all and (if multiple triggers fire)
    which tag to embed in the filename — first match wins, in the
    order of priority documented in the spec.
    """
    triggers: List[str] = []
    fk = str(failure_kind or "")
    if cfg.on_nonfinite_state and fk == "nonfinite_state":
        triggers.append("nonfinite_state")
    if cfg.on_invalid_input and fk == "invalid_input":
        triggers.append("invalid_input")
    if (cfg.on_residual_nan
            and not (np.isfinite(residual) if residual is not None
                     else False)):
        triggers.append("residual_nan")
    if cfg.on_converged_false and not bool(converged):
        triggers.append("converged_false")
    if (cfg.on_n_inner_at_max
            and int(n_inner) >= int(n_inner_max)):
        triggers.append("budget")
    if cfg.on_force_nan and (
            (F_hyd_x is not None and not np.isfinite(F_hyd_x))
            or (F_hyd_y is not None and not np.isfinite(F_hyd_y))):
        triggers.append("force_nan")
    return triggers


# ─── Filename ──────────────────────────────────────────────────────


def build_dump_filename(
    *,
    step: int,
    substep: int,
    trial: int,
    commit: bool,
    primary_trigger: str,
) -> str:
    """``ausas_call_step{step:04d}_sub{sub}_trial{trial:03d}_commit{0|1}_{trigger}.npz``"""
    s_step = max(int(step), -1)
    s_sub = max(int(substep), -1)
    s_trial = max(int(trial), -1)
    return (
        f"ausas_call_step{s_step:04d}"
        f"_sub{s_sub}"
        f"_trial{s_trial:03d}"
        f"_commit{1 if commit else 0}"
        f"_{primary_trigger}.npz"
    )


# ─── Writer ────────────────────────────────────────────────────────


def write_dump_npz(
    out_dir: str,
    fname: str,
    payload: Dict[str, Any],
) -> str:
    """Write ``payload`` to ``out_dir/fname`` via numpy.savez.

    Object-typed entries (lists, dicts, str) round-trip through
    numpy.asarray with ``dtype=object``. Returns the full path
    on success.
    """
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, fname)
    coerced: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            coerced[k] = np.array("None", dtype=object)
        elif isinstance(v, np.ndarray):
            coerced[k] = v
        elif isinstance(v, (str, bool, int, float)):
            coerced[k] = np.asarray(v)
        elif isinstance(v, (list, tuple)):
            coerced[k] = np.asarray(v, dtype=object)
        elif isinstance(v, dict):
            coerced[k] = np.asarray([v], dtype=object)
        else:
            coerced[k] = np.asarray(v, dtype=object)
    np.savez(target, **coerced)
    return target


def validate_dump_npz(path: str) -> Tuple[bool, List[str]]:
    """Open ``path`` and verify all ``MANDATORY_DUMP_KEYS`` are
    present. Returns ``(ok, missing)`` — ``ok=True`` means every
    mandatory key is in the file.

    Used by the test suite and by the spec-mandated
    ``--validate-dumps`` flag (out of scope for this commit; the
    helper is exposed so a follow-up CLI can call it).
    """
    if not os.path.exists(path):
        return False, list(MANDATORY_DUMP_KEYS)
    with np.load(path, allow_pickle=True) as nz:
        present = set(nz.files)
    missing = [k for k in MANDATORY_DUMP_KEYS if k not in present]
    return (not missing), missing


__all__ = [
    "MANDATORY_DUMP_KEYS",
    "OPTIONAL_FORCE_INPUT_KEYS",
    "DumpConfig",
    "DumpCounters",
    "evaluate_triggers",
    "build_dump_filename",
    "write_dump_npz",
    "validate_dump_npz",
]
