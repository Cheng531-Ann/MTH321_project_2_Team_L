# Code Map

| Area | Main files |
| --- | --- |
| Model and fitted operator | `stage1_validation.py`, `local_vol_model.py` |
| LCP solver and diagnostics | `stage1_validation.py` |
| Stage 2 local volatility | `stage2_local_volatility.py`, `stage2_convergence_report.py` |
| Independent ODE reference | `reference_solution.py` |
| Mesh construction | `grids.py` |
| Solver comparison and tolerance sweep | `experiment_solvers.py` |
| Adaptive/Robin/adjoint extensions | `advanced_methods.py`, `local_vol_adjoint.py` |
| Two-dimensional spread put | `stage2d_spread.py`, `stage2d_nine_point.py` |
| Plotting only | `plot_*.py` |
| End-to-end reproduction | `run_all.py` |

`run_all.py` executes the experiments and plotting scripts in dependency order,
checks that every required output exists, and synchronises the published
copies into `../03_Figures` and `../04_Data`.

