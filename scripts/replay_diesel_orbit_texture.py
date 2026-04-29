#!/usr/bin/env python
"""Stage J fu-2 Task 4 — prescribed-orbit replay for texture screening.

Replays a previously-recorded smooth orbit through Ausas with a
*different* texture, with no Verlet mechanics and no Picard
coupling. Cavitation state is threaded between steps so the result
is mass-conserving on the replay grid; orbit is fixed.

Used as a cheap surrogate (×10–×30 faster than full transient) to
rank texture candidates before committing to coupled runs.

Pure I/O around the existing
``models.diesel_ausas_adapter.ausas_one_step_with_state`` plus the
force/friction helpers in ``models.diesel_transient``. No GPU
solver invocation other than that single backend call per step.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import diesel_params as params  # noqa: E402
from config.diesel_groove_presets import (  # noqa: E402
    resolve_groove_preset,
)
from models.bearing_model import setup_grid, setup_texture  # noqa: E402
from models.diesel_ausas_adapter import (  # noqa: E402
    DieselAusasState,
    ausas_one_step_with_state,
)
from models.diesel_transient import (  # noqa: E402
    CONFIGS,
    CONFIG_KEYS,
    build_H_2d,
    compute_friction,
    compute_hydro_forces,
)
from models.groove_geometry import (  # noqa: E402
    build_herringbone_groove_relief,
)


__all__ = ["replay_run", "main"]


# ─── Source-data loading ───────────────────────────────────────────


def _find_data_npz(run_dir: str) -> str:
    for fname in ("data.npz", "data_partial.npz"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No data.npz / data_partial.npz in {run_dir!r}")


def _resolve_source_index(npz: Dict[str, Any], label: str) -> int:
    """Map a label or CONFIG_KEYS alias to the row index in the
    saved per-config arrays."""
    labels = [str(s) for s in np.asarray(npz["labels"]).tolist()]
    if label in labels:
        return labels.index(label)
    if label in CONFIG_KEYS:
        target_label = CONFIGS[CONFIG_KEYS[label]]["label"]
        if target_label in labels:
            return labels.index(target_label)
    raise SystemExit(
        f"--source-config {label!r} not found among run labels "
        f"{labels} (or CONFIG_KEYS aliases {sorted(CONFIG_KEYS)}).")


@dataclass
class SourceOrbit:
    """Last-cycle slice from the source run, packaged for replay."""
    phi_deg: np.ndarray
    eps_x: np.ndarray
    eps_y: np.ndarray
    eta_eff: np.ndarray
    dt_s: np.ndarray
    n_phi: int
    n_z: int
    omega: float
    source_label: str
    pmax_source: np.ndarray
    P_loss_source: np.ndarray
    hmin_source: np.ndarray


def _load_source_orbit(
    run_dir: str, source_label: str,
) -> Tuple[SourceOrbit, Dict[str, Any]]:
    """Load the last cycle of the source config from data.npz."""
    data_path = _find_data_npz(run_dir)
    npz = dict(np.load(data_path, allow_pickle=True))
    ic = _resolve_source_index(npz, source_label)
    last_start = int(npz.get("last_start", 0))
    n_real = int(np.asarray(npz.get(
        "steps_completed", np.array([npz["pmax"].shape[1]]))
    )[ic])
    if n_real <= last_start:
        raise SystemExit(
            f"Source config has only {n_real} completed steps; "
            f"last_start={last_start} — no full last cycle to replay.")
    sl = slice(last_start, n_real)
    phi_full = np.asarray(npz["phi_crank_deg"], dtype=float)[sl]
    eps_x = np.asarray(npz["eps_x"], dtype=float)[ic, sl]
    eps_y = np.asarray(npz["eps_y"], dtype=float)[ic, sl]
    eta_eff = np.asarray(npz["eta_eff"], dtype=float)[ic, sl]
    # dt_s reconstructed from successive crank-angle differences;
    # pad the last entry with the previous one (the runner's
    # last step has no "next" angle, but we never integrate past it).
    n = len(phi_full)
    if n >= 2:
        dphi = np.diff(phi_full)
        dphi = np.concatenate([dphi, dphi[-1:]])
    else:
        dphi = np.array([1.0])
    n_rpm = float(getattr(params, "n", 1900.0))
    omega = 2.0 * np.pi * n_rpm / 60.0
    dt_s = np.deg2rad(dphi) / omega
    n_phi = int(np.asarray(npz["pmax"]).shape[1])  # not the grid; ignored
    # Real grid width comes from saved scalars.
    n_phi = int(npz.get("N_phi_grid", n_phi))
    n_z = int(npz.get("N_z_grid", n_phi))
    pmax_src = np.asarray(npz["pmax"], dtype=float)[ic, sl]
    P_loss_src = np.asarray(npz["P_loss"], dtype=float)[ic, sl]
    hmin_src = np.asarray(npz["hmin"], dtype=float)[ic, sl]
    return SourceOrbit(
        phi_deg=phi_full,
        eps_x=eps_x, eps_y=eps_y, eta_eff=eta_eff,
        dt_s=dt_s,
        n_phi=int(n_phi), n_z=int(n_z),
        omega=omega,
        source_label=source_label,
        pmax_source=pmax_src,
        P_loss_source=P_loss_src,
        hmin_source=hmin_src,
    ), npz


# ─── Target texture builder ────────────────────────────────────────


def _build_target_texture(
    target_cfg: Dict[str, Any],
    *,
    Phi_mesh: np.ndarray, Z_mesh: np.ndarray,
    texture_kind: str,
    groove_preset: Optional[str],
    n_phi: int, n_z: int,
):
    """Return (textured, texture_kind, groove_relief, phi_c, Z_c)
    for ``build_H_2d`` calls during replay."""
    if not target_cfg.get("textured", False):
        return False, "none", None, None, None
    if texture_kind == "groove":
        if groove_preset is None:
            raise SystemExit(
                "--texture-kind=groove requires --groove-preset.")
        pr = resolve_groove_preset(
            str(groove_preset),
            R_m=float(params.R), L_m=float(params.L),
            c_m=float(params.c),
        )
        relief = build_herringbone_groove_relief(
            Phi_mesh, Z_mesh,
            variant=pr["variant"],
            depth_nondim=pr["depth_nondim"],
            N_branch_per_side=pr["N_branch_per_side"],
            w_branch_nondim=pr["w_branch_nondim"],
            belt_half_nondim=pr["belt_half_nondim"],
            beta_deg=pr["beta_deg"],
            ramp_frac=pr["ramp_frac"],
            taper_ratio=pr["taper_ratio"],
            apex_radius_frac=pr["apex_radius_frac"],
            chirality=pr["chirality"],
            coverage_mode=pr["coverage_mode"],
            protected_lo_deg=pr["protected_lo_deg"],
            protected_hi_deg=pr["protected_hi_deg"],
        )
        return True, "groove", relief, None, None
    # Dimple — legacy elliptical pockets.
    phi_c, Z_c = setup_texture(params)
    return True, "dimple", None, phi_c, Z_c


@dataclass
class ReplayStepRow:
    """One per-step row, written to CSV / used by summary."""
    step: int
    phi: float
    eps_x: float
    eps_y: float
    p_max: float
    h_min: float
    F_hyd_x: float
    F_hyd_y: float
    P_loss: float
    cav_frac: float
    n_inner: int
    residual: float
    converged: bool


# ─── Per-target replay ─────────────────────────────────────────────


def _replay_one_target(
    src: SourceOrbit,
    target_label: str,
    *,
    n_phi: int, n_z: int,
    texture_kind: str,
    groove_preset: Optional[str],
    ausas_tol: float,
    ausas_max_inner: int,
) -> List[ReplayStepRow]:
    """Run the single-pass Ausas replay over the source orbit for
    one target config. Returns per-step diagnostic rows."""
    target_cfg = CONFIGS[CONFIG_KEYS[target_label]]
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(
        n_phi, n_z)
    textured, tex_kind, relief, phi_c, Z_c = _build_target_texture(
        target_cfg,
        Phi_mesh=Phi_mesh, Z_mesh=Z_mesh,
        texture_kind=texture_kind, groove_preset=groove_preset,
        n_phi=n_phi, n_z=n_z,
    )
    # Stage J Bug 5 — initialise state from the first accepted gap
    # so step #1 doesn't see an artificial squeeze impulse.
    H0 = build_H_2d(
        src.eps_x[0], src.eps_y[0], Phi_mesh, Z_mesh, params,
        textured=textured, texture_kind=tex_kind,
        groove_relief=relief, phi_c_flat=phi_c, Z_c_flat=Z_c)
    state = DieselAusasState.from_initial_gap(H0)
    extra = dict(tol=float(ausas_tol),
                  max_inner=int(ausas_max_inner))

    rows: List[ReplayStepRow] = []
    for k in range(len(src.phi_deg)):
        eps_x_t = float(src.eps_x[k])
        eps_y_t = float(src.eps_y[k])
        eta_t = float(src.eta_eff[k])
        dt_s_t = float(src.dt_s[k])
        H_curr = build_H_2d(
            eps_x_t, eps_y_t, Phi_mesh, Z_mesh, params,
            textured=textured, texture_kind=tex_kind,
            groove_relief=relief, phi_c_flat=phi_c, Z_c_flat=Z_c)
        result = ausas_one_step_with_state(
            state,
            H_curr=H_curr,
            dt_s=dt_s_t,
            omega_shaft=src.omega,
            d_phi=float(d_phi), d_Z=float(d_Z),
            R=float(params.R), L=float(params.L),
            extra_options=extra,
            commit=True,
        )
        p_scale = (6.0 * eta_t * src.omega
                   * (params.R / params.c) ** 2)
        if result.converged and result.P_nd is not None:
            Fx, Fy = compute_hydro_forces(
                result.P_nd, p_scale, Phi_mesh, phi_1D, Z_1D,
                params.R, params.L)
            F_friction = compute_friction(
                result.P_nd, p_scale, H_curr,
                Phi_mesh, phi_1D, Z_1D,
                eta_t, src.omega,
                params.R, params.L, params.c)
            U = src.omega * params.R
            P_loss_t = float(F_friction * U)
            p_max_t = float(np.max(result.P_nd) * p_scale)
            cav_frac_t = float(
                np.mean(result.theta < 1.0)
                if result.theta is not None else 0.0)
        else:
            Fx = Fy = float("nan")
            P_loss_t = float("nan")
            p_max_t = float("nan")
            cav_frac_t = float("nan")
        rows.append(ReplayStepRow(
            step=k,
            phi=float(src.phi_deg[k]),
            eps_x=eps_x_t, eps_y=eps_y_t,
            p_max=p_max_t,
            h_min=float(np.min(H_curr) * params.c),
            F_hyd_x=float(Fx), F_hyd_y=float(Fy),
            P_loss=P_loss_t,
            cav_frac=cav_frac_t,
            n_inner=int(result.n_inner),
            residual=float(result.residual),
            converged=bool(result.converged),
        ))
    return rows


# ─── Output writers ────────────────────────────────────────────────


_CSV_FIELDS = [
    "target_config", "step", "phi_deg",
    "eps_x", "eps_y",
    "p_max_Pa", "h_min_m",
    "F_hyd_x_N", "F_hyd_y_N",
    "P_loss_W", "cav_frac",
    "n_inner", "residual", "converged",
]


def _write_csv(
    out_path: str,
    rows_per_target: Dict[str, List[ReplayStepRow]],
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for label, rows in rows_per_target.items():
            for r in rows:
                w.writerow({
                    "target_config": label,
                    "step": r.step,
                    "phi_deg": r.phi,
                    "eps_x": r.eps_x, "eps_y": r.eps_y,
                    "p_max_Pa": r.p_max, "h_min_m": r.h_min,
                    "F_hyd_x_N": r.F_hyd_x, "F_hyd_y_N": r.F_hyd_y,
                    "P_loss_W": r.P_loss, "cav_frac": r.cav_frac,
                    "n_inner": r.n_inner, "residual": r.residual,
                    "converged": int(r.converged),
                })


def _aggregates(rows: List[ReplayStepRow]) -> Dict[str, float]:
    """Per-target last-cycle aggregates. NaN-safe."""
    if not rows:
        return {}
    pmax = np.array([r.p_max for r in rows], dtype=float)
    hmin = np.array([r.h_min for r in rows], dtype=float)
    Ploss = np.array([r.P_loss for r in rows], dtype=float)
    cavf = np.array([r.cav_frac for r in rows], dtype=float)
    n_conv = int(sum(1 for r in rows if r.converged))
    return dict(
        n_steps=len(rows),
        n_converged=n_conv,
        p_max_max=float(np.nanmax(pmax)) if pmax.size else float("nan"),
        p_max_mean=float(np.nanmean(pmax)) if pmax.size else float("nan"),
        h_min_min=float(np.nanmin(hmin)) if hmin.size else float("nan"),
        P_loss_mean=(float(np.nanmean(Ploss))
                      if Ploss.size else float("nan")),
        P_loss_total=(float(np.nansum(Ploss))
                       if Ploss.size else float("nan")),
        cav_frac_mean=(float(np.nanmean(cavf))
                        if cavf.size else float("nan")),
    )


def _write_summary(
    out_path: str,
    src: SourceOrbit,
    rows_per_target: Dict[str, List[ReplayStepRow]],
) -> None:
    lines: List[str] = []
    lines.append(
        "Stage J fu-2 Task 4 — replay (prescribed orbit, "
        "qualitative screening)")
    lines.append(f"Source config       : {src.source_label}")
    lines.append(
        f"Last-cycle steps    : {len(src.phi_deg)} "
        f"(phi {src.phi_deg[0]:.1f}° → {src.phi_deg[-1]:.1f}°)")
    lines.append(f"Replay grid         : N_phi={src.n_phi}, "
                  f"N_z={src.n_z}")
    lines.append(f"omega_shaft         : {src.omega:.3f} rad/s")
    lines.append("")
    src_aggs = dict(
        p_max_max=float(np.nanmax(src.pmax_source))
        if src.pmax_source.size else float("nan"),
        p_max_mean=float(np.nanmean(src.pmax_source))
        if src.pmax_source.size else float("nan"),
        h_min_min=float(np.nanmin(src.hmin_source))
        if src.hmin_source.size else float("nan"),
        P_loss_mean=float(np.nanmean(src.P_loss_source))
        if src.P_loss_source.size else float("nan"),
    )
    lines.append("source aggregates (full transient, last cycle):")
    lines.append(
        f"  p_max max/mean      : {src_aggs['p_max_max']/1e6:.2f} / "
        f"{src_aggs['p_max_mean']/1e6:.3f} MPa")
    lines.append(
        f"  h_min min           : "
        f"{src_aggs['h_min_min']*1e6:.2f} µm")
    lines.append(
        f"  P_loss mean         : "
        f"{src_aggs['P_loss_mean']:.1f} W")
    lines.append("")
    for label, rows in rows_per_target.items():
        a = _aggregates(rows)
        lines.append(f"target: {label}")
        lines.append(
            f"  n_steps / converged : {a['n_steps']} / "
            f"{a['n_converged']}")
        lines.append(
            f"  p_max max/mean      : "
            f"{a['p_max_max']/1e6:.2f} / "
            f"{a['p_max_mean']/1e6:.3f} MPa")
        lines.append(
            f"  h_min min           : "
            f"{a['h_min_min']*1e6:.2f} µm")
        lines.append(
            f"  P_loss mean / sum   : "
            f"{a['P_loss_mean']:.1f} W / "
            f"{a['P_loss_total']:.1f} W·step")
        lines.append(
            f"  cav_frac mean       : {a['cav_frac_mean']:.3f}")
        # Compare against source (only meaningful if target == source).
        if label == src.source_label:
            try:
                rel_err_pmax = (np.abs(
                    np.array([r.p_max for r in rows]) - src.pmax_source)
                    / np.maximum(np.abs(src.pmax_source), 1e3))
                rel_err_Ploss = (np.abs(
                    np.array([r.P_loss for r in rows])
                    - src.P_loss_source)
                    / np.maximum(np.abs(src.P_loss_source), 1.0))
                lines.append(
                    "  vs source (smooth→smooth self-consistency):")
                lines.append(
                    f"    median rel err p_max  : "
                    f"{float(np.nanmedian(rel_err_pmax)):.3%}")
                lines.append(
                    f"    median rel err P_loss : "
                    f"{float(np.nanmedian(rel_err_Ploss)):.3%}")
            except (ValueError, FloatingPointError):
                pass
        lines.append("")
    lines.append(
        "NOTE: replay (prescribed orbit, no Verlet/Picard) is a "
        "qualitative screening — use full transient for production "
        "claims.")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_plots(
    out_dir: str,
    rows_per_target: Dict[str, List[ReplayStepRow]],
) -> None:
    """Four PNGs (pmax / hmin / Nloss / cavfrac vs phi). Quietly
    skipped if matplotlib isn't available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    series = [
        ("replay_pmax_phi.png", "p_max [MPa]",
         lambda r: r.p_max / 1e6),
        ("replay_hmin_phi.png", "h_min [µm]",
         lambda r: r.h_min * 1e6),
        ("replay_Nloss_phi.png", "P_loss [W]", lambda r: r.P_loss),
        ("replay_cavfrac_phi.png", "cav_frac", lambda r: r.cav_frac),
    ]
    for fname, ylabel, ext in series:
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, rows in rows_per_target.items():
            phi = np.array([r.phi for r in rows], dtype=float)
            y = np.array([ext(r) for r in rows], dtype=float)
            ax.plot(phi, y, label=label, lw=1.0)
        ax.set_xlabel("crank angle [deg]")
        ax.set_ylabel(ylabel)
        ax.set_title(
            "replay (prescribed orbit, qualitative screening)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)


# ─── Top-level entry ───────────────────────────────────────────────


def replay_run(
    *,
    source_data: str,
    source_config: str,
    target_configs: List[str],
    texture_kind: str,
    groove_preset: Optional[str],
    n_phi: Optional[int],
    n_z: Optional[int],
    out_dir: str,
    ausas_tol: float,
    ausas_max_inner: int,
    write_plots: bool = True,
) -> Dict[str, List[ReplayStepRow]]:
    """Pure-function replay; returns per-target row dicts and
    writes outputs to ``out_dir``. Side effects only on disk."""
    src, _npz = _load_source_orbit(source_data, source_config)
    if n_phi is not None:
        src.n_phi = int(n_phi)
    if n_z is not None:
        src.n_z = int(n_z)
    os.makedirs(out_dir, exist_ok=True)
    rows_per_target: Dict[str, List[ReplayStepRow]] = {}
    for tgt in target_configs:
        if tgt not in CONFIG_KEYS:
            raise SystemExit(
                f"--target-configs entry {tgt!r} unknown; valid: "
                f"{sorted(CONFIG_KEYS)}")
        rows_per_target[tgt] = _replay_one_target(
            src, tgt,
            n_phi=src.n_phi, n_z=src.n_z,
            texture_kind=texture_kind,
            groove_preset=groove_preset,
            ausas_tol=ausas_tol,
            ausas_max_inner=ausas_max_inner,
        )
    _write_csv(os.path.join(out_dir, "replay_metrics.csv"),
                 rows_per_target)
    _write_summary(os.path.join(out_dir, "replay_summary.txt"),
                     src, rows_per_target)
    if write_plots:
        _write_plots(out_dir, rows_per_target)
    return rows_per_target


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage J fu-2 Task 4 — prescribed-orbit "
                    "replay for cheap texture screening.")
    p.add_argument("--source-data", required=True,
                    help="Path to a transient-run directory (with "
                         "data.npz / data_partial.npz).")
    p.add_argument("--source-config", default="mineral_smooth",
                    help="CONFIG_KEYS alias OR exact label of the "
                         "source config (default: mineral_smooth).")
    p.add_argument("--target-configs",
                    default="mineral_smooth,mineral_textured",
                    help="Comma-separated CONFIG_KEYS aliases.")
    p.add_argument("--texture-kind", default="groove",
                    choices=["dimple", "groove", "none"])
    p.add_argument("--groove-preset", default=None)
    p.add_argument("--n-phi", type=int, default=None)
    p.add_argument("--n-z", type=int, default=None)
    p.add_argument("--out-dir", default=None,
                    help="Output directory (default: "
                         "<source-data>/replay/).")
    p.add_argument("--ausas-tol", type=float, default=1e-4)
    p.add_argument("--ausas-max-inner", type=int, default=5000)
    p.add_argument("--no-plots", action="store_true", default=False)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = (args.out_dir
                if args.out_dir
                else os.path.join(args.source_data, "replay"))
    targets = [s.strip() for s in str(args.target_configs).split(",")
                if s.strip()]
    replay_run(
        source_data=args.source_data,
        source_config=args.source_config,
        target_configs=targets,
        texture_kind=args.texture_kind,
        groove_preset=args.groove_preset,
        n_phi=args.n_phi, n_z=args.n_z,
        out_dir=out_dir,
        ausas_tol=args.ausas_tol,
        ausas_max_inner=args.ausas_max_inner,
        write_plots=not args.no_plots,
    )
    print(f"[replay] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
