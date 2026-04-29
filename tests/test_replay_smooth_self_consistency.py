"""Stage J fu-2 Task 4 — replay self-consistency contract.

Runs the prescribed-orbit replay against a synthetic source npz
with target=source. The fake Ausas backend is deterministic in
H, so for the same orbit replay reproduces the source's per-step
metrics bit-for-bit (well within the engineering 5% / 10% bound
the spec asks for).

The test exercises:

1. ``_load_source_orbit`` — last-cycle slice extraction;
2. ``_replay_one_target`` — orbit fixed, state threaded;
3. ``_write_csv`` / ``_write_summary`` — output writers don't crash;
4. round-trip pmax / P_loss agreement vs the synthetic source
   numbers stored in the npz.

No real GPU. No matplotlib (``--no-plots`` equivalent).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.diesel_ausas_adapter import (  # noqa: E402
    set_ausas_backend_for_tests,
)


# ─── Deterministic fake Ausas backend ──────────────────────────────


def _fake_ausas_backend(**kwargs):
    """Single-shot fake — returns a deterministic pressure field
    derived from H. The precise formula is irrelevant; what matters
    is that the same H produces the same P (so replay → source).

    Uses ``**kwargs`` to accept whatever the adapter passes
    (``H_curr`` / ``H_prev`` / ``dt`` / ``d_phi`` / ``d_Z`` / etc.)
    without coupling to the real solver's exact signature. Stage J
    Bug 3 ghost-grid migration sends padded ``(N_z, N_phi+2)`` H
    arrays — we operate on those shapes and trust the adapter to
    unpad."""
    H = np.asarray(kwargs["H_curr"], dtype=float)
    # Pressure is a smooth function of (H, phi-index): inversely
    # proportional to H, modulated by sin(phi-index). Stays finite
    # and positive for any plausible H.
    n_z, n_phi_padded = H.shape
    phi_grid = np.arange(n_phi_padded)[None, :]
    P = (1.0 / np.maximum(H, 1e-3)) * (
        0.5 + 0.4 * np.sin(2.0 * np.pi * phi_grid / n_phi_padded))
    theta = np.where(P > 1.5, 0.7 * np.ones_like(P),
                       np.ones_like(P))
    # Canonical 4-tuple: (P, theta, residual, n_inner)
    return (P, theta, 1e-7, 100)


@pytest.fixture(autouse=True)
def _install_fake_backend():
    set_ausas_backend_for_tests(_fake_ausas_backend)
    try:
        yield
    finally:
        set_ausas_backend_for_tests(None)


# ─── Source npz builder ────────────────────────────────────────────


def _build_synthetic_source_npz(tmp_path):
    """Build a minimal data.npz that ``_load_source_orbit`` can
    parse. The orbit is a small ellipse; the per-step metrics are
    populated by RUNNING the replay with target=source first
    (so synthetic numbers match what the fake backend would
    produce). That makes self-consistency a tautology in the
    sense that we're testing the replay LOGIC, not a real solver."""
    from scripts.replay_diesel_orbit_texture import _replay_one_target

    n_phi = 16
    n_z = 8
    # Two cycles of 30 steps each (60 total). last_start=30.
    n_steps = 60
    last_start = 30
    dphi_per_step = 720.0 / n_steps  # ~12°/step (legacy-isotropic)
    phi_crank_deg = np.arange(n_steps) * dphi_per_step

    # Smooth orbit: small ellipse, ε ∈ ~0.3.
    t = np.linspace(0.0, 4.0 * np.pi, n_steps)
    eps_x_full = 0.30 * np.cos(t)
    eps_y_full = 0.20 * np.sin(t)
    eta_eff_full = np.full(n_steps, 0.04, dtype=float)

    # Build the source npz with placeholder per-step results, then
    # immediately compute the per-step results by running the
    # replay with target=source on this orbit. Write a SECOND npz
    # that has the matching pmax/P_loss/hmin entries — that's the
    # source data the test will actually replay against.
    source_dir_stage1 = tmp_path / "stage1"
    source_dir_stage1.mkdir()
    n_cfg = 1
    placeholder = np.zeros((n_cfg, n_steps), dtype=float)
    np.savez(
        os.path.join(str(source_dir_stage1), "data.npz"),
        labels=np.array(["mineral_smooth"], dtype=object),
        phi_crank_deg=phi_crank_deg,
        last_start=int(last_start),
        n_steps_per_cycle=int(n_steps - last_start),
        steps_completed=np.array([n_steps], dtype=np.int32),
        steps_attempted=np.array([n_steps], dtype=np.int32),
        eps_x=eps_x_full[None, :],
        eps_y=eps_y_full[None, :],
        eta_eff=eta_eff_full[None, :],
        pmax=placeholder.copy(),
        P_loss=placeholder.copy(),
        hmin=placeholder.copy(),
        N_phi_grid=int(n_phi),
        N_z_grid=int(n_z),
        ausas_options=np.asarray(
            [{"tol": 1e-6, "max_inner": 5000}], dtype=object),
    )

    # First replay — use it to populate the source's per-step
    # metrics, so the round-trip comparison is meaningful.
    from scripts.replay_diesel_orbit_texture import _load_source_orbit
    src, _ = _load_source_orbit(str(source_dir_stage1),
                                  "mineral_smooth")
    rows = _replay_one_target(
        src, "mineral_smooth",
        n_phi=n_phi, n_z=n_z,
        texture_kind="none", groove_preset=None,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    # Project the last-cycle rows back into a full per-step array
    # (steps before last_start get zero / NaN so the source-side
    # last-cycle slice picks up exactly the rows produced).
    pmax_full = np.full(n_steps, float("nan"))
    Ploss_full = np.full(n_steps, float("nan"))
    hmin_full = np.full(n_steps, float("nan"))
    for k, r in enumerate(rows):
        idx = last_start + k
        pmax_full[idx] = r.p_max
        Ploss_full[idx] = r.P_loss
        hmin_full[idx] = r.h_min

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    np.savez(
        os.path.join(str(source_dir), "data.npz"),
        labels=np.array(["mineral_smooth"], dtype=object),
        phi_crank_deg=phi_crank_deg,
        last_start=int(last_start),
        n_steps_per_cycle=int(n_steps - last_start),
        steps_completed=np.array([n_steps], dtype=np.int32),
        steps_attempted=np.array([n_steps], dtype=np.int32),
        eps_x=eps_x_full[None, :],
        eps_y=eps_y_full[None, :],
        eta_eff=eta_eff_full[None, :],
        pmax=pmax_full[None, :],
        P_loss=Ploss_full[None, :],
        hmin=hmin_full[None, :],
        N_phi_grid=int(n_phi),
        N_z_grid=int(n_z),
        ausas_options=np.asarray(
            [{"tol": 1e-6, "max_inner": 5000}], dtype=object),
    )
    return str(source_dir), n_phi, n_z


# ─── Tests ─────────────────────────────────────────────────────────


def test_replay_smooth_to_smooth_self_consistency(tmp_path):
    """target=source replay reproduces source per-step metrics on
    the last cycle within engineering tolerance (5% median, 10% p99).
    """
    from scripts.replay_diesel_orbit_texture import replay_run

    src_dir, n_phi, n_z = _build_synthetic_source_npz(tmp_path)
    out_dir = tmp_path / "replay_out"
    rows_per_target = replay_run(
        source_data=src_dir,
        source_config="mineral_smooth",
        target_configs=["mineral_smooth"],
        texture_kind="none", groove_preset=None,
        n_phi=n_phi, n_z=n_z,
        out_dir=str(out_dir),
        ausas_tol=1e-6, ausas_max_inner=5000,
        write_plots=False,
    )
    rows = rows_per_target["mineral_smooth"]
    # Re-load source numbers for the comparison.
    npz = dict(np.load(os.path.join(src_dir, "data.npz"),
                        allow_pickle=True))
    last_start = int(npz["last_start"])
    n_real = int(npz["steps_completed"][0])
    pmax_src = np.asarray(npz["pmax"])[0, last_start:n_real]
    Ploss_src = np.asarray(npz["P_loss"])[0, last_start:n_real]

    pmax_replay = np.array([r.p_max for r in rows])
    Ploss_replay = np.array([r.P_loss for r in rows])
    rel_err_pmax = (np.abs(pmax_replay - pmax_src)
                     / np.maximum(np.abs(pmax_src), 1e3))
    rel_err_Ploss = (np.abs(Ploss_replay - Ploss_src)
                     / np.maximum(np.abs(Ploss_src), 1.0))

    # Median ≤ 5% / p99 ≤ 10% — the spec target.
    assert float(np.nanmedian(rel_err_pmax)) < 0.05
    assert float(np.nanmedian(rel_err_Ploss)) < 0.05
    assert float(np.nanpercentile(rel_err_pmax, 99)) < 0.10
    assert float(np.nanpercentile(rel_err_Ploss, 99)) < 0.10


def test_replay_writes_csv_and_summary(tmp_path):
    """Output files exist after the run."""
    from scripts.replay_diesel_orbit_texture import replay_run

    src_dir, n_phi, n_z = _build_synthetic_source_npz(tmp_path)
    out_dir = tmp_path / "out2"
    replay_run(
        source_data=src_dir,
        source_config="mineral_smooth",
        target_configs=["mineral_smooth"],
        texture_kind="none", groove_preset=None,
        n_phi=n_phi, n_z=n_z,
        out_dir=str(out_dir),
        ausas_tol=1e-6, ausas_max_inner=5000,
        write_plots=False,
    )
    assert (out_dir / "replay_metrics.csv").exists()
    assert (out_dir / "replay_summary.txt").exists()
    txt = (out_dir / "replay_summary.txt").read_text(encoding="utf-8")
    assert "Stage J fu-2 Task 4" in txt
    assert "mineral_smooth" in txt


def test_replay_unknown_source_config(tmp_path):
    from scripts.replay_diesel_orbit_texture import replay_run
    src_dir, n_phi, n_z = _build_synthetic_source_npz(tmp_path)
    with pytest.raises(SystemExit):
        replay_run(
            source_data=src_dir,
            source_config="not_a_real_label",
            target_configs=["mineral_smooth"],
            texture_kind="none", groove_preset=None,
            n_phi=n_phi, n_z=n_z,
            out_dir=str(tmp_path / "out3"),
            ausas_tol=1e-6, ausas_max_inner=5000,
            write_plots=False,
        )


def test_replay_unknown_target_config(tmp_path):
    from scripts.replay_diesel_orbit_texture import replay_run
    src_dir, n_phi, n_z = _build_synthetic_source_npz(tmp_path)
    with pytest.raises(SystemExit):
        replay_run(
            source_data=src_dir,
            source_config="mineral_smooth",
            target_configs=["bogus_target"],
            texture_kind="none", groove_preset=None,
            n_phi=n_phi, n_z=n_z,
            out_dir=str(tmp_path / "out4"),
            ausas_tol=1e-6, ausas_max_inner=5000,
            write_plots=False,
        )
