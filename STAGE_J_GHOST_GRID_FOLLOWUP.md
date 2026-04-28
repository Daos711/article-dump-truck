# Stage J followup-2 Step 12 — ghost-grid contract mismatch

**Status:** open. Recorded by `tests/test_diesel_ausas_ghost_grid_contract.py::test_ausas_adapter_phi_ghost_grid_contract` (currently `@pytest.mark.xfail(strict=True)`).

## Mismatch

The expert-mandated pipeline-side contract requires:

| Surface | Shape & responsibility |
|---|---|
| `DieselAusasState.{P, theta, H_prev}` | physical `(N_z, N_phi)` |
| `ausas_one_step_with_state` input args | physical `(N_z, N_phi)` |
| **adapter → backend kwargs** | **padded `(N_z, N_phi+2)` with `seam[:, 0]=phys[:, -1]`, `seam[:, -1]=phys[:, 0]`** |
| backend → adapter return | padded `(N_z, N_phi+2)` |
| `ausas_one_step_with_state` return + commit | physical `(N_z, N_phi)` |
| explicit kwargs to backend | `periodic_phi=True`, `periodic_z=False`, `p_bc_z0=p_bc_zL=0`, `theta_bc_z0=theta_bc_zL=1` |

The current adapter (`models/diesel_ausas_adapter.py`, "Bug 3" fix) does **not** pad. It ships unpadded `(N_z, N_phi)` to `ausas_unsteady_one_step_gpu`, which performs `_pack_ghosts(..., periodic_phi=True)` internally on the solver side. The docstring on `pad_phi_for_ausas` and the ctor of `DieselAusasState` explicitly warn that pre-padding here triggers a **double-wrap** that corrupts every step.

The two architectures are mutually exclusive: only one of {adapter, solver} may own the padding.

## Why this matters

Adapter-side padding (the expert-mandated contract) is the cleaner pipeline boundary — it makes the adapter the single locus of pad/unpad knowledge, lets the solver assume "your input is already padded with seam ghosts", and removes the runtime coupling between `extra_options` and solver-internal `_pack_ghosts` semantics. It also unblocks future PV-on-Ausas backends (Stage K) that may not want a hard-wired `_pack_ghosts(periodic_phi=True)` call.

## Open questions for the expert

1. **Migration scope.** Should the fix land as part of Step 12 (this PR) or as a separate adapter / solver migration PR after Stage J fu-2 closes?
2. **Solver-side coordination.** Disabling solver-side `_pack_ghosts` requires a change in `reynolds_solver.cavitation.ausas.solver_dynamic_gpu.ausas_unsteady_one_step_gpu` (likely a `seam_ghosts_already_packed=True` kwarg). Who owns that change? Is the kwarg already in flight in `reynolds_solver`?
3. **Acceptance order.** Do we need to re-run the GPU smoke (Gate 2) and bit-for-bit Gate 1 invariance after the migration? Both legacy_verlet HS path and damped Ausas path touch this code.
4. **`d_phi` semantics.** Should `d_phi` stay at the physical spacing `2π / N_phi` (test asserts this), or does the solver expect `2π / (N_phi + 2)` after the migration? Current test pins the physical value.

## Proposed plan (subject to expert review)

1. Add `seam_ghosts_already_packed: bool = False` kwarg to `ausas_unsteady_one_step_gpu`. When `True`, skip the internal `_pack_ghosts` call.
2. In `models/diesel_ausas_adapter.py:ausas_one_step_with_state`:
   * Pad all four input arrays via `pad_phi_for_ausas` immediately before the backend call.
   * Pass `seam_ghosts_already_packed=True` in the kwargs.
   * Pass `periodic_phi=True`, `periodic_z=False`, `p_bc_z0=p_bc_zL=0`, `theta_bc_z0=theta_bc_zL=1` explicitly (defaults already match but explicit-is-better for the contract).
   * Unpad the backend's return arrays via `unpad_phi_from_ausas` before storing in `DieselAusasState` / `DieselAusasStepResult`.
3. Update `DieselAusasState` docstring: `P`/`theta`/`H_prev` stay on physical `(N_z, N_phi)`; the adapter handles all padding.
4. Remove the "Bug 3" warnings from `pad_phi_for_ausas` / `unpad_phi_from_ausas` / `DieselAusasState` docstrings.
5. Drop `@pytest.mark.xfail` from `test_ausas_adapter_phi_ghost_grid_contract`.
6. Re-run Gate 1 (legacy invariance, HS path — should be unaffected, HS doesn't use the Ausas adapter) and Gate 2 (smoke, damped Ausas path — must reach φ ≥ 120°).

## What's needed from the user

* Approval of the plan above (or replacement plan).
* Confirmation on solver-side coordination: is `seam_ghosts_already_packed` (or equivalent) already available in `reynolds_solver`, or does this require a separate solver-side PR?
* Final text for this document — current content is the agent's best-effort summary of the mismatch; the expert prescribed minimal content that should be substituted in.
