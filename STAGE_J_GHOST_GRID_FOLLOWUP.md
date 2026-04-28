# Stage J followup-2 Step 12 — ghost-grid contract mismatch

**Status:** open. Recorded by
`tests/test_diesel_ausas_ghost_grid_contract.py::test_ausas_adapter_phi_ghost_grid_contract`,
currently marked `xfail(strict=True)`.

## Contract

Diesel pipeline state is physical / endpoint-free in the circumferential
direction:

| Object | Shape |
|---|---|
| `DieselAusasState.P` | `(N_z, N_phi)` |
| `DieselAusasState.theta` | `(N_z, N_phi)` |
| `DieselAusasState.H_prev` | `(N_z, N_phi)` |
| `ausas_one_step_with_state(H_curr=...)` input | `(N_z, N_phi)` |

The Ausas dynamic GPU solver uses one seam ghost column on each
circumferential side. Therefore the adapter/backend boundary must be:

| Boundary object | Shape / convention |
|---|---|
| backend `H_curr` | `(N_z, N_phi + 2)` |
| backend `H_prev` | `(N_z, N_phi + 2)` |
| backend `P_prev` | `(N_z, N_phi + 2)` |
| backend `theta_prev` | `(N_z, N_phi + 2)` |
| left phi ghost | `padded[:, 0] = physical[:, -1]` |
| right phi ghost | `padded[:, -1] = physical[:, 0]` |
| backend return `P`, `theta` | padded `(N_z, N_phi + 2)` |
| adapter return / commit | physical `(N_z, N_phi)` |

`d_phi` remains the physical spacing:

```python
d_phi = 2π / N_phi
```

It must not be computed from the padded width.

The adapter must also pass the boundary convention explicitly:

```
periodic_phi = True
periodic_z = False
p_bc_z0 = p_bc_zL = 0.0
theta_bc_z0 = theta_bc_zL = 1.0
```

## Observed mismatch

The current adapter sends unpadded `(N_z, N_phi)` arrays directly to
`ausas_unsteady_one_step_gpu`.

The solver-side dynamic Ausas kernel treats column 0 and column -1
as seam ghost columns when `periodic_phi=True`; the interior physical
columns are `1:-1`. As a result, passing an endpoint-free physical
diesel grid causes the first and last real diesel columns to be treated
as ghost cells, and the effective physical circumferential grid becomes
`N_phi - 2`.

This is the mismatch captured by the xfailed contract test.

## Solver-side note

The current solver-side `_pack_ghosts(...)` routine is an in-place ghost
refresh:

```python
arr[:, 0] = arr[:, N_phi - 2]
arr[:, -1] = arr[:, 1]
```

It does not allocate a second ghost layer. Therefore adapter-side padding
to `(N_z, N_phi + 2)` is compatible with the current solver: `_pack_ghosts`
will simply refresh the seam ghosts.

No `seam_ghosts_already_packed` kwarg exists in the current solver
archive. A solver-side PR is not required for the Stage J adapter
migration unless we choose to add an explicit API guard later.

## Migration plan

1. In `models/diesel_ausas_adapter.ausas_one_step_with_state`:
   * keep state arrays physical;
   * pad `H_curr`, `H_prev`, `P_prev`, and `theta_prev` immediately before
     the backend call;
   * keep `d_phi = 2π / N_phi_physical`;
   * pass `periodic_phi=True`, `periodic_z=False`, and the Z boundary
     values explicitly;
   * do not pass unknown solver kwargs such as `seam_ghosts_already_packed`
     unless the solver-side API actually supports them;
   * unpad backend `P` and `theta` before returning or committing.
2. Update docstrings that currently claim the adapter must not pre-pad.
3. Remove `xfail(strict=True)` from
   `test_ausas_adapter_phi_ghost_grid_contract`.
4. Update `tests/test_diesel_ausas_real_backend.py` so Bug 3 means:
   physical state/result, padded backend boundary.
5. Re-run:
   * adapter unit tests;
   * ghost-grid contract test;
   * real backend smoke if GPU/CuPy is available;
   * Gate 1 legacy invariance;
   * Gate 2 damped Ausas smoke to at least φ >= 120°.

## Acceptance

The migration is accepted when:

* `DieselAusasState` remains physical `(N_z, N_phi)`;
* backend kwargs are padded `(N_z, N_phi + 2)`;
* returned `P_nd` / `theta` are physical `(N_z, N_phi)`;
* `d_phi` is still `2π / N_phi`;
* the xfailed ghost-grid test becomes a normal passing test;
* Gate 1 remains invariant;
* Gate 2 Ausas smoke passes the agreed early-cycle threshold.
