"""Diagnostic C contract tests (coexp_v1.1).

Проверяется только инфраструктура:
  1. --Wy-share-list создаёт отдельные подкаталоги equilibrium_wy<XXX>/
  2. --tol-accept-override фиксируется в manifest как tol_accept_effective
  3. compute_decision_signal корректен на mock-данных

Тесты не требуют reynolds_solver (conftest ставит стабы) и не требуют
реальных solve — используется subprocess-runner, который подменяет
`solve_equilibrium_pair` через monkeypatch в рантайме.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.coexp_schema import SCHEMA_VERSION  # noqa: E402

# diagnostic-mode schema bump
DIAGNOSTIC_SCHEMA = "coexp_v1.1"


# ── Helpers ────────────────────────────────────────────────────────

def _write_top6(base: str, useful_profile_ids):
    """Synthesize confirm/top6_confirmed.json with 3 candidates."""
    import hashlib
    cf = os.path.join(base, "confirm")
    os.makedirs(cf, exist_ok=True)
    cands = []
    for i, pid in enumerate(useful_profile_ids):
        cands.append(dict(
            profile_id=pid, family="two_lobe",
            profile_spec=dict(family="two_lobe",
                              params={"A2": 0.05, "phi2_deg": 10.0 + i}),
            J_screen=0.1 - 0.01 * i,
        ))
    with open(os.path.join(cf, "top6_confirmed.json"), "w") as f:
        json.dump(dict(
            schema_version=SCHEMA_VERSION,
            top_n=len(cands), score_key="J_screen",
            candidates=cands,
        ), f)


def _runner_script(tmp_path, base_results, run_id, from_confirm, extra_argv,
                    useful_ids_per_wy):
    """Create a runner that:
      * puts tests/conftest stubs on sys.path so reynolds_solver is stubbed
      * monkeypatches solve_equilibrium_pair to skip real PS solve
      * monkeypatches results path to tmp
      * invokes run_coexp_equilibrium.main()
    """
    runner = tmp_path / "run_eq_runner.py"
    code = f"""
import sys, os, json
sys.path.insert(0, {ROOT!r})
sys.path.insert(0, {os.path.join(ROOT, 'scripts')!r})
sys.path.insert(0, {os.path.join(ROOT, 'tests')!r})

# install reynolds_solver stub from conftest machinery
import conftest  # side effect: sys.modules['reynolds_solver*']

# Patch results path: run_coexp_equilibrium.main() builds `base` from
# os.path.join(__file__, '..', 'results', 'coexp'). Override by passing
# through env-ish shim: patch os.path.dirname for that call? Simpler —
# monkeypatch `os.path.join` is overkill. Instead we patch the
# `results/coexp` location by prepending a symlink-like path via
# creating the expected dir layout under tmp AND chdir-ing.

os.chdir({str(tmp_path)!r})

# Build the tmp results tree pre-run
tmp_root = {str(tmp_path)!r}
results_coexp = os.path.join(tmp_root, "results", "coexp")
os.makedirs(results_coexp, exist_ok=True)

# Link the confirm/top6_confirmed.json into expected path:
# the script reads os.path.join(os.path.dirname(__file__), "..",
# "results", "coexp", <from_confirm>, "confirm", "top6_confirmed.json").
# We can't easily rewrite __file__; instead, monkeypatch the script's
# main() to use our tmp as results root. Simplest: monkeypatch
# os.path.join behaviour for "results"/"coexp" segment.

# Actually simpler: we copy pre-made top6 into the path THE SCRIPT
# expects (relative to its own __file__). The script uses
#   base = os.path.join(os.path.dirname(__file__), "..", "results",
#                        "coexp")
# where __file__ is the actual script path under ROOT. So artifacts
# will be written to ROOT/results/coexp/... — NOT tmp. We DON'T want
# that (it pollutes repo). Fix: monkeypatch
# run_coexp_equilibrium.base via direct attribute set after import but
# before main().

import importlib.util
spec_path = os.path.join({ROOT!r}, 'scripts', 'run_coexp_equilibrium.py')
spec = importlib.util.spec_from_file_location('run_coexp_equilibrium', spec_path)
mod = importlib.util.module_from_spec(spec)
sys.modules['run_coexp_equilibrium'] = mod
spec.loader.exec_module(mod)

# Monkeypatch solve_equilibrium_pair to return canned pairs.
from models.magnetic_equilibrium import EquilibriumResult, result_status
useful_ids_per_wy = {useful_ids_per_wy!r}

def _fake_solve(experiment, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                 R, L, c, sigma, eta, omega, F0, Wy_share,
                 ps_solver, texture_relief_fn=None,
                 tol_accept=5e-3, step_cap=0.1, eps_max=0.9,
                 max_iter=80):
    profile_hash = experiment.profile_id
    wy_key = f"{{Wy_share:.3f}}"
    is_useful = profile_hash in useful_ids_per_wy.get(wy_key, [])
    # Compose canned result dict mirroring the real function's output.
    base_metric = dict(
        X=0.0, Y=-0.3, eps=0.30, attitude_deg=0.0,
        Fx_hydro=0.0, Fy_hydro=0.0, Fx_mag=0.0, Fy_mag=0.0,
        h_min=20.0e-6, p_max=6.0e6, cav_frac=0.05,
        friction=1.0,
        rel_residual=1e-4, n_iter=5, converged=True,
        unload_share_target=0.0, unload_share_actual=0.0,
        hydro_share_actual=1.0, K_mag=0.0, status='accepted',
    )
    sm = dict(base_metric)
    if is_useful:
        tx = dict(base_metric, h_min=20.3e-6, p_max=5.9e6,
                  friction=1.01, cav_frac=0.055, eps=0.29)
    else:
        tx = dict(base_metric, h_min=19.5e-6, p_max=6.1e6,
                  friction=1.03, cav_frac=0.07, eps=0.31)
    return dict(
        experiment_id=experiment.experiment_id,
        profile_hash=profile_hash,
        profile_spec=experiment.profile.as_dict(),
        smooth=sm,
        textured=tx,
        smooth_accepted=True,
        textured_accepted=True,
    )

mod.solve_equilibrium_pair = _fake_solve

# Redirect results/coexp base to tmp by patching os.path.dirname
# result of __file__ inside the module. Simpler — we directly
# patch the `base` compute using a small wrapper on `main`:
_orig_main = mod.main
import argparse
_orig_parse_args = argparse.ArgumentParser.parse_args
def _patched_parse_args(self):
    ns = _orig_parse_args(self)
    return ns
argparse.ArgumentParser.parse_args = _patched_parse_args

# Overwrite os.path functions isn't needed — we just pre-create the
# top6 under the script's expected path. run-id is {run_id!r}, so we
# will write into ROOT/results/coexp/<run_id>/ and then move outputs
# to tmp afterwards. To keep the repo clean for tests, we set up
# a sandboxed ROOT for this run.

# Approach: monkeypatch `os.path.dirname(__file__)` inside the module
# via setting module attribute and forcing main to re-resolve paths.
# Cleanest: monkeypatch the `os.path.join(...) => base` expression by
# patching `os.makedirs` to NOT actually make dirs outside tmp.
# But the script uses absolute paths — we just make sure they live
# under tmp.

# Shim: replace module-level os.path.dirname(__file__) for the script
# to resolve under tmp.
# Simpler: monkey-patch `os.path.abspath` for this specific shim.

# Actually the cleanest: set an env override read by the script is not
# present. So we'll physically create a symlink tree: ROOT/results →
# tmp/results. Since we can't write under ROOT in tests, we just
# override the `base` computation by patching the module's `main`.

def patched_main():
    # Force base to tmp-root
    import run_coexp_equilibrium as m
    orig_join = os.path.join
    def tmp_join(*parts):
        # intercept the specific 4-arg pattern "dirname(__file__), '..',
        # 'results', 'coexp'"
        if (len(parts) >= 4 and parts[-2] == 'results'
                and parts[-1] == 'coexp'):
            return os.path.join(tmp_root, 'results', 'coexp')
        return orig_join(*parts)
    os.path.join = tmp_join
    try:
        _orig_main()
    finally:
        os.path.join = orig_join

# Pre-populate confirm/top6 in tmp
from tests_stub_helpers import make_confirm_top6  # noqa: F401

sys.argv = [spec_path, '--run-id', {run_id!r},
            '--from-confirm', {from_confirm!r}] + {extra_argv!r}
patched_main()

# Emit marker for test harness
print('RUNNER_OK')
"""
    runner.write_text(code)
    return runner


def _make_confirm_top6(base_results_coexp, from_confirm, profile_ids):
    confirm_dir = os.path.join(base_results_coexp, from_confirm, "confirm")
    os.makedirs(confirm_dir, exist_ok=True)
    cands = [
        dict(profile_id=pid, family="two_lobe",
             profile_spec=dict(family="two_lobe",
                               params={"A2": 0.05, "phi2_deg": 10.0 + i}),
             J_screen=0.1 - 0.01 * i)
        for i, pid in enumerate(profile_ids)
    ]
    with open(os.path.join(confirm_dir, "top6_confirmed.json"), "w") as f:
        json.dump(dict(
            schema_version=SCHEMA_VERSION,
            top_n=len(cands),
            score_key="J_screen",
            candidates=cands,
        ), f)


def _ensure_stub_helpers():
    path = os.path.join(ROOT, "tests", "tests_stub_helpers.py")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                "def make_confirm_top6(*a, **kw):\n"
                "    pass\n")


# ── T1: --Wy-share-list creates separate subdirs ────────────────────

def test_wy_share_list_creates_separate_subdirs(tmp_path):
    _ensure_stub_helpers()
    run_id = "diagrun"
    from_confirm = "diagrun"
    base_coexp = tmp_path / "results" / "coexp"
    _make_confirm_top6(str(base_coexp), from_confirm,
                       profile_ids=["abc123def4560001",
                                    "abc123def4560002"])
    runner = _runner_script(
        tmp_path, str(base_coexp), run_id, from_confirm,
        extra_argv=["--Wy-share-list", "0.05,0.10,0.25",
                    "--grid", "16x8",
                    "--tol-accept-override", "0.01",
                    "--max-iter-nr", "200"],
        useful_ids_per_wy={"0.050": [], "0.100": [], "0.250": []},
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert "RUNNER_OK" in r.stdout, (
        f"runner failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")

    for tag in ("005", "010", "025"):
        subdir = base_coexp / run_id / f"equilibrium_wy{tag}"
        assert subdir.is_dir(), f"missing subdir {subdir}"
        summary_path = subdir / "equilibrium_summary.json"
        assert summary_path.exists(), f"missing {summary_path}"
        with open(summary_path) as f:
            s = json.load(f)
        assert s["schema_version"] == DIAGNOSTIC_SCHEMA
        # Each Wy summary carries its own Wy_share
        assert s["Wy_share"] == pytest.approx(int(tag) / 100.0)


# ── T2: --tol-accept-override recorded in manifest ──────────────────

def test_tol_accept_override_recorded_in_manifest(tmp_path):
    _ensure_stub_helpers()
    run_id = "diagrun2"
    from_confirm = "diagrun2"
    base_coexp = tmp_path / "results" / "coexp"
    _make_confirm_top6(str(base_coexp), from_confirm,
                       profile_ids=["abc123def4560001"])
    runner = _runner_script(
        tmp_path, str(base_coexp), run_id, from_confirm,
        extra_argv=["--Wy-share-list", "0.05",
                    "--grid", "16x8",
                    "--tol-accept-override", "0.01",
                    "--max-iter-nr", "200"],
        useful_ids_per_wy={"0.050": []},
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert "RUNNER_OK" in r.stdout, r.stdout + r.stderr

    subdir = base_coexp / run_id / "equilibrium_wy005"
    summary_path = subdir / "equilibrium_summary.json"
    manifest_path = subdir / "manifest.json"
    with open(summary_path) as f:
        s = json.load(f)
    with open(manifest_path) as f:
        m = json.load(f)
    # Both must record effective tol and max_iter
    assert s["tol_accept_effective"] == pytest.approx(0.01)
    assert m["tol_accept_effective"] == pytest.approx(0.01)
    assert s["max_iter_nr"] == 200
    assert m["max_iter_nr"] == 200
    # Baseline tol_accept field (raw CLI default) MUST NOT silently
    # override the effective value
    assert s["tol_accept"] == pytest.approx(0.01)


# ── T3: summary decision_signal is correct on mock data ────────────

def test_summary_diagnostic_C_decision_signal():
    from run_coexp_diagnostic_C import compute_decision_signal

    # Case PROCEED: 1 useful at low-load Wy=0.10
    by_wy = {
        "0.050": [dict(profile_hash="a", useful=False),
                   dict(profile_hash="b", useful=False)],
        "0.100": [dict(profile_hash="a", useful=True),
                   dict(profile_hash="b", useful=False)],
        "0.250": [dict(profile_hash="a", useful=False),
                   dict(profile_hash="b", useful=False)],
    }
    assert compute_decision_signal(
        by_wy, low_load_keys=["0.050", "0.100"]) == "PROCEED_TO_V2"

    # Case PROCEED: useful only at Wy=0.05
    by_wy = {
        "0.050": [dict(profile_hash="a", useful=True)],
        "0.100": [dict(profile_hash="b", useful=False)],
        "0.250": [dict(profile_hash="c", useful=False)],
    }
    assert compute_decision_signal(
        by_wy, low_load_keys=["0.050", "0.100"]) == "PROCEED_TO_V2"

    # Case CLOSE: nothing useful anywhere
    by_wy = {
        "0.050": [dict(profile_hash="a", useful=False)],
        "0.100": [dict(profile_hash="b", useful=False)],
        "0.250": [dict(profile_hash="c", useful=False)],
    }
    assert compute_decision_signal(
        by_wy, low_load_keys=["0.050", "0.100"]) == "CLOSE_COEXP"

    # Case CLOSE: useful only at control Wy=0.25 — doesn't count
    by_wy = {
        "0.050": [dict(profile_hash="a", useful=False)],
        "0.100": [dict(profile_hash="b", useful=False)],
        "0.250": [dict(profile_hash="c", useful=True)],
    }
    assert compute_decision_signal(
        by_wy, low_load_keys=["0.050", "0.100"]) == "CLOSE_COEXP"

    # Edge: empty low_load keys
    assert compute_decision_signal({}, low_load_keys=[]) == "CLOSE_COEXP"


# ── Bonus: E2 + Wy-share-list не совместимы ────────────────────────

def test_E2_plus_Wy_share_list_fails(tmp_path):
    _ensure_stub_helpers()
    run_id = "diagrun3"
    from_confirm = "diagrun3"
    base_coexp = tmp_path / "results" / "coexp"
    _make_confirm_top6(str(base_coexp), from_confirm,
                       profile_ids=["abc123def4560001"])

    # Just run the script with conflict args — expect non-zero exit and
    # an informative message on stderr/stdout. We don't even need the
    # full runner; we import main() and catch SystemExit.
    import importlib.util
    spec_path = os.path.join(ROOT, "scripts", "run_coexp_equilibrium.py")
    spec = importlib.util.spec_from_file_location(
        "run_coexp_equilibrium_import", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Force args via sys.argv
    sys.argv = [spec_path,
                 "--run-id", run_id,
                 "--from-confirm", from_confirm,
                 "--Wy-share-list", "0.05,0.10",
                 "--grid", "16x8",
                 "--phase", "E2"]
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code != 0
