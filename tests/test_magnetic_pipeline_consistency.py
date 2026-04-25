"""Контрактные тесты magnetic pipeline (ТЗ §6).

Эти тесты проверяют, что:
 1. smooth manifest содержит и baseline_raw, и baseline_canonical;
    baseline_canonical accepted на target=0.0
 2. textured compare НЕ пересчитывает smooth — вызывает find_equilibrium
    только для textured cases
 3. compare читает tol_accept из manifest, а не hardcode
 4. pairs.unload_share_target бит-в-бит совпадает со smooth_accepted
 5. plot/schema isolation: legacy JSON → plot падает

Тесты не требуют reynolds_solver. Мы стабим зависимости через
sys.modules, monkeypatch-им find_equilibrium и запускаем compare
как Python-скрипт в subprocess с пробросом PYTHONPATH на stub-shim.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.magnetic_equilibrium import (  # noqa: E402
    EquilibriumResult, is_accepted, result_to_dict, result_status,
)


# ───────────────────────── fixtures ──────────────────────────────

def make_result(X, Y, eps, target, K_mag, res=1e-4, converged=True,
                unload=0.0, hydro=1.0, h_min=1e-5, p_max=1e7):
    r = EquilibriumResult(
        X=X, Y=Y, eps=eps, attitude_deg=0.0,
        Fx_hydro=0.0, Fy_hydro=0.0, Fx_mag=0.0, Fy_mag=0.0,
        h_min=h_min, p_max=p_max, cav_frac=0.0, friction=0.0,
        rel_residual=res, n_iter=5, converged=converged,
        unload_share_target=float(target),
        unload_share_actual=unload, hydro_share_actual=hydro,
        K_mag=K_mag,
    )
    r.status = result_status(r, 5e-3)
    return r


def write_manifest(run_dir, targets=(0.0, 0.025, 0.05, 0.10),
                   tol_accept=5e-3, step_cap=0.10, eps_max=0.90):
    os.makedirs(run_dir, exist_ok=True)
    baseline_raw = make_result(0.0, -0.4, 0.40, 0.0, 0.0, res=6e-2)
    baseline_raw.converged = True
    baseline_raw.rel_residual = 6e-2  # намеренно раздутый — NOT accepted
    baseline_raw.status = result_status(baseline_raw, tol_accept)

    # accepted smooth entries (unique K_mag per target)
    smooth_accepted = []
    smooth_all = []
    for t in targets:
        X = 0.0
        Y = -0.4 + 0.1 * t / 0.1
        eps = (X * X + Y * Y) ** 0.5
        K = 0.0 if t == 0.0 else 10.0 * t
        r = make_result(X, Y, eps, t, K, res=1e-4,
                        unload=t, hydro=1.0 - t)
        smooth_accepted.append(result_to_dict(r))
        smooth_all.append(dict(
            unload_share_target=float(t),
            result=result_to_dict(r),
            accepted=True,
        ))

    baseline_canonical = smooth_accepted[0]

    manifest = {
        "schema_version": "magnetic_v4",
        "run_id": "test_run",
        "model": "radial",
        "created_utc": "2026-04-16T00:00:00Z",
        "config": {
            "N_phi": 64,
            "N_Z": 32,
            "W_applied_N": [0.0, -9956.7],
            "F0_N": 39826.85,
            "p_scale_Pa": 2.0e7,
            "tol_accept": float(tol_accept),
            "step_cap": float(step_cap),
            "eps_max": float(eps_max),
            "targets": [float(t) for t in targets],
        },
        "baseline_raw": result_to_dict(baseline_raw),
        "baseline_canonical": baseline_canonical,
        "smooth_accepted": smooth_accepted,
        "smooth_all": smooth_all,
        "acceptance": {
            "baseline_reproduced": True,
            "unload_positive": True,
            "eps_monotonic": True,
            "max_residual": 1e-4,
            "sum_shares_ok": True,
            "accepted_targets": [float(t) for t in targets],
        },
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path, manifest


# ─────────────── Test 1: canonical baseline ──────────────────────

def test_canonical_baseline_exists_and_accepted(tmp_path):
    run_dir = tmp_path / "run1"
    mp, mn = write_manifest(str(run_dir))
    with open(mp) as f:
        m = json.load(f)
    # оба baseline поля есть
    assert "baseline_raw" in m
    assert "baseline_canonical" in m
    assert m["baseline_raw"] is not None
    assert m["baseline_canonical"] is not None

    # canonical attached к target=0.0
    assert m["baseline_canonical"]["unload_share_target"] == 0.0

    # canonical accepted под manifest.config.tol_accept
    tol = m["config"]["tol_accept"]
    assert m["baseline_canonical"]["rel_residual"] <= tol, (
        f"canonical rel_residual={m['baseline_canonical']['rel_residual']} "
        f"> tol_accept={tol}")
    assert m["baseline_canonical"].get("status") == "accepted"


def test_canonical_differs_from_raw_when_raw_not_accepted(tmp_path):
    """Если baseline_raw НЕ accepted, canonical должен быть другой и accepted.

    Это именно та ситуация, которую мы хотим лечить (см. ТЗ §2.1).
    """
    run_dir = tmp_path / "run2"
    mp, mn = write_manifest(str(run_dir))
    with open(mp) as f:
        m = json.load(f)
    tol = m["config"]["tol_accept"]
    assert m["baseline_raw"]["rel_residual"] > tol  # setup
    # canonical должен иметь другой rel_residual (и быть принят)
    assert m["baseline_canonical"]["rel_residual"] < tol


# ─────────────── Test 2: compare does not recompute smooth ───────
#
# Стратегия: запускаем run_mag_textured_compare.py как subprocess,
# подкладывая:
#   - stub reynolds_solver (PS solver + create_H_with_ellipsoidal_depressions
#     возвращают простые тензоры)
#   - обёртку find_equilibrium, которая логирует каждый вызов и его
#     "target" по характеру H_and_force (textured или smooth)
# Если смотрим на то, какие `H_and_force` функции были переданы в
# find_equilibrium: make_H_and_force_textured (см. compare) помечена
# атрибутом `__is_textured__ = True` через monkeypatch.
#
# Простая альтернатива: use sys.modules stub, install find_equilibrium
# counter, fail on smooth invocation.

STUB_REYNOLDS = """
import types, sys, numpy as np

def _fake_ps(H, d_phi, d_Z, R, L, tol=1e-6, max_iter=100):
    P = np.zeros_like(H)
    theta = np.ones_like(H)
    return P, theta, None, None

def _fake_hde(H0, depth, Phi, Zm, phi_c, Z_c, a_Z, a_phi, profile=None):
    return np.asarray(H0, dtype=float) - 1e-6

reynolds_solver = types.ModuleType('reynolds_solver')
cav = types.ModuleType('reynolds_solver.cavitation')
ps = types.ModuleType('reynolds_solver.cavitation.payvar_salant')
utils = types.ModuleType('reynolds_solver.utils')

ps.solve_payvar_salant_gpu = _fake_ps
ps.solve_payvar_salant_cpu = _fake_ps
utils.create_H_with_ellipsoidal_depressions = _fake_hde

sys.modules['reynolds_solver'] = reynolds_solver
sys.modules['reynolds_solver.cavitation'] = cav
sys.modules['reynolds_solver.cavitation.payvar_salant'] = ps
sys.modules['reynolds_solver.utils'] = utils
"""


def _make_compare_runner(tmp_path, manifest_path,
                          log_path, extra_pre=""):
    """Generate a runner.py that:
      1) installs reynolds_solver stubs
      2) monkeypatches models.magnetic_equilibrium.find_equilibrium so that:
         - each call logs (is_textured_marker, tol, target_hint) to log_path
         - returns a pre-fabricated EquilibriumResult
      3) invokes compare.main() with --manifest <path>
    """
    runner = tmp_path / "runner.py"
    code = f"""
import sys, os, json
sys.path.insert(0, {ROOT!r})

{STUB_REYNOLDS}

{extra_pre}

import models.magnetic_equilibrium as meq
from models.magnetic_equilibrium import EquilibriumResult, result_status

CALL_LOG = []
_orig = meq.find_equilibrium

def fake_find(H_and_force, mag_model, W_applied,
              X0=0.0, Y0=-0.4, max_iter=80, tol=1e-4,
              step_cap=0.10, eps_max=0.90, tol_accept=None):
    marker = getattr(H_and_force, '__kind__', 'unknown')
    CALL_LOG.append(dict(kind=marker, tol=float(tol),
                          tol_accept=(float(tol_accept)
                                      if tol_accept is not None else None),
                          X0=float(X0), Y0=float(Y0),
                          K_mag=float(mag_model.scale)))
    r = EquilibriumResult(
        X=float(X0), Y=float(Y0),
        eps=float((X0*X0+Y0*Y0)**0.5),
        attitude_deg=0.0,
        Fx_hydro=0.0, Fy_hydro=0.0, Fx_mag=0.0, Fy_mag=0.0,
        h_min=1.0e-5, p_max=1.0e7, cav_frac=0.0, friction=1.0,
        rel_residual=1.0e-4, n_iter=1, converged=True,
        K_mag=float(mag_model.scale),
    )
    r.status = result_status(r, float(tol_accept if tol_accept else tol))
    return r

meq.find_equilibrium = fake_find

# Импорт compare-модуля. Переопределяем make_H_and_force_textured,
# чтобы передать маркер __kind__ на возвращаемом closure.
import importlib.util
spec = importlib.util.spec_from_file_location(
    'compare_mod',
    os.path.join({ROOT!r}, 'scripts', 'run_mag_textured_compare.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['compare_mod'] = mod
# meq.find_equilibrium уже подменён; compare использует его через
# from ... import find_equilibrium, поэтому также подменим атрибут
# модуля после его загрузки.
spec.loader.exec_module(mod)
mod.find_equilibrium = fake_find  # ensure compare sees fake

# Маркер textured
_orig_factory = mod.make_H_and_force_textured
def wrapped_factory(*a, **kw):
    f = _orig_factory(*a, **kw)
    f.__kind__ = 'textured'
    return f
mod.make_H_and_force_textured = wrapped_factory

# main() читает sys.argv
sys.argv = ['compare', '--manifest', {str(manifest_path)!r}]
mod.main()

with open({str(log_path)!r}, 'w', encoding='utf-8') as f:
    json.dump(CALL_LOG, f)
"""
    runner.write_text(code)
    return runner


def test_compare_does_not_recompute_smooth(tmp_path):
    run_dir = tmp_path / "run"
    manifest_path, m = write_manifest(str(run_dir))

    log_path = tmp_path / "calls.json"
    runner = _make_compare_runner(tmp_path, manifest_path, log_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, str(runner)],
                        capture_output=True, text=True, env=env)
    assert r.returncode == 0, (
        "compare runner failed:\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")

    with open(log_path) as f:
        calls = json.load(f)

    # Все вызовы find_equilibrium должны иметь kind='textured'.
    kinds = {c["kind"] for c in calls}
    assert kinds == {"textured"}, (
        f"compare called find_equilibrium with kinds={kinds!r}; "
        f"expected only 'textured'")
    # И должно быть минимум по одному на accepted smooth (fallback может
    # добавить повторы, но не меньше len(accepted)).
    n_accepted = len(m["smooth_accepted"])
    assert len(calls) >= n_accepted


# ─────────────── Test 3: tolerance consistency ───────────────────

def test_compare_reads_tol_accept_from_manifest(tmp_path):
    run_dir = tmp_path / "run"
    manifest_path, m = write_manifest(
        str(run_dir), tol_accept=7.3e-3, step_cap=0.07, eps_max=0.85)

    log_path = tmp_path / "calls.json"
    runner = _make_compare_runner(tmp_path, manifest_path, log_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, str(runner)],
                        capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"stderr:\n{r.stderr}"

    with open(log_path) as f:
        calls = json.load(f)

    # Каждый вызов fake_find должен получить tol и tol_accept из manifest.
    for c in calls:
        assert c["tol_accept"] == pytest.approx(7.3e-3), c
        assert c["tol"] == pytest.approx(7.3e-3), c


# ─────────────── Test 4: accepted target consistency ─────────────

def test_compare_pairs_match_manifest_accepted_targets(tmp_path):
    run_dir = tmp_path / "run"
    manifest_path, m = write_manifest(str(run_dir))

    log_path = tmp_path / "calls.json"
    runner = _make_compare_runner(tmp_path, manifest_path, log_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, str(runner)],
                        capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"stderr:\n{r.stderr}"

    out_json = os.path.join(os.path.dirname(manifest_path),
                             "textured_compare.json")
    assert os.path.exists(out_json), "compare не записал textured_compare.json"
    with open(out_json) as f:
        tx = json.load(f)

    assert tx["schema_version"] == "magnetic_v4"
    pair_targets = [p["unload_share_target"] for p in tx["pairs"]]
    smooth_targets = [s["unload_share_target"] for s in m["smooth_accepted"]]
    assert pair_targets == smooth_targets, (
        f"compare pairs targets {pair_targets!r} != "
        f"manifest accepted {smooth_targets!r}")

    # smooth_ref в каждой паре бит-в-бит совпадает со smooth_accepted
    for p, s in zip(tx["pairs"], m["smooth_accepted"]):
        assert p["smooth_ref"] == s, (
            "smooth_ref в compare пairs отличается от manifest smooth_accepted")


# ─────────────── Test 5: plot schema isolation ───────────────────

def test_plot_rejects_legacy_json(tmp_path):
    """plot_mag_results.py должен падать на legacy JSON (ТЗ §4.4.3)."""
    run_dir = tmp_path / "legacy_run"
    os.makedirs(run_dir)
    # Legacy-shaped manifest.json без schema_version magnetic_v4
    legacy = {
        "mag_share_target": [0.0, 0.05, 0.10],
        "continuation": [{"mag_share_target": 0.0, "eps": 0.4}],
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(legacy, f)

    plot_script = os.path.join(ROOT, "scripts", "plot_mag_results.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, plot_script, "--data-dir", str(run_dir)],
        capture_output=True, text=True, env=env)
    # должен упасть с ненулевым кодом
    assert r.returncode != 0, (
        f"plot не упал на legacy JSON. stdout:\n{r.stdout}\nstderr:\n"
        f"{r.stderr}")
    # В выводе должно быть упоминание schema или legacy
    combined = (r.stdout + r.stderr).lower()
    assert ("schema" in combined or "legacy" in combined or
            "magnetic_v4" in combined), (
        f"plot упал, но не из-за schema check:\n{r.stdout}\n{r.stderr}")


def test_plot_accepts_v4_manifest_without_textured(tmp_path):
    """Если есть только manifest.json (без textured_compare.json),
    plot должен отработать и НЕ генерировать ratios график
    (ТЗ §4.4.5)."""
    # skip если matplotlib недоступен
    try:
        import matplotlib  # noqa: F401
    except Exception:
        pytest.skip("matplotlib not available")

    run_dir = tmp_path / "v4_run"
    write_manifest(str(run_dir))

    plot_script = os.path.join(ROOT, "scripts", "plot_mag_results.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, plot_script, "--data-dir", str(run_dir)],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, (
        f"plot не смог построить базовый отчёт:\n{r.stdout}\n{r.stderr}")
    # Нет textured pairs → нет ratios.png
    assert not os.path.exists(run_dir / "ratios_tex_vs_smooth.png"), (
        "plot сгенерировал ratios plot без accepted textured pairs")
    # report.md существует
    assert os.path.exists(run_dir / "report.md")


# ─────────────── Test: helper contract ──────────────────────────

def test_is_accepted_none_returns_false():
    assert is_accepted(None, 1e-3) is False


def test_is_accepted_respects_converged_flag():
    r = make_result(0, 0, 0, 0, 0, res=1e-5, converged=False)
    assert is_accepted(r, 1e-3) is False


def test_is_accepted_respects_tol():
    r = make_result(0, 0, 0, 0, 0, res=4e-3, converged=True)
    assert is_accepted(r, 5e-3) is True
    assert is_accepted(r, 1e-3) is False


def test_result_status_classification():
    # accepted
    r1 = make_result(0, 0, 0, 0, 0, res=1e-4, converged=True)
    assert result_status(r1, 5e-3) == "accepted"
    # soft_converged
    r2 = make_result(0, 0, 0, 0, 0, res=1e-2, converged=True)
    assert result_status(r2, 5e-3) == "soft_converged"
    # failed
    r3 = make_result(0, 0, 0, 0, 0, res=1e-2, converged=False)
    assert result_status(r3, 5e-3) == "failed"
    assert result_status(None, 5e-3) == "failed"
