"""Local-volatility adjoint sensitivities for the perpetual American put.

The constant-volatility adjoint experiment is validated against a closed-form
derivative in ``advanced_methods.py``.  This module extends the same discrete
adjoint construction to the smooth local-volatility profile used in Stage 2.
No closed-form derivative is available there, so the check is a direct
finite-difference solve of the full LCP at several target prices.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from advanced_methods import (
    NonuniformOperator,
    build_nonuniform_operator,
    interior_matrix,
    solve_operator,
    uniform_grid,
)
from local_vol_model import K, Q, R, S_MAX, local_volatility


TARGETS = (80.0, 90.0, 100.0, 110.0, 120.0, 150.0)
TOL_SLACK = 1e-10 * K
PARAMETERS = ('scale', 'r', 'q')


def _sigma(scale: float):
    """Return the local-volatility profile multiplied by ``scale``."""
    return lambda S: scale * local_volatility(S, K_value=K)


def _operator(
    S: np.ndarray,
    scale: float = 1.0,
    r: float = R,
    q: float = Q,
) -> NonuniformOperator:
    return build_nonuniform_operator(
        S,
        K_value=K,
        r=r,
        q=q,
        sigma=_sigma(scale),
        boundary='dirichlet',
    )


def _parameter_value(parameter: str) -> float:
    if parameter == 'scale':
        return 1.0
    if parameter == 'r':
        return R
    if parameter == 'q':
        return Q
    raise ValueError(f'unknown parameter: {parameter}')


def _operator_at(S: np.ndarray, parameter: str, value: float) -> NonuniformOperator:
    scale, r, q = 1.0, R, Q
    if parameter == 'scale':
        scale = value
    elif parameter == 'r':
        r = value
    elif parameter == 'q':
        q = value
    else:
        raise ValueError(f'unknown parameter: {parameter}')
    return _operator(S, scale=scale, r=r, q=q)


def _interpolation_weights(S: np.ndarray, target: float) -> np.ndarray:
    """Weights for linear interpolation of an interior grid vector."""
    n_int = S.size - 2
    weights = np.zeros(n_int)
    upper = int(np.searchsorted(S, target))
    upper = min(max(upper, 1), S.size - 1)
    lower = upper - 1
    fraction = (target - S[lower]) / (S[upper] - S[lower])
    if lower >= 1:
        weights[lower - 1] += 1.0 - fraction
    if upper <= S.size - 2:
        weights[upper - 1] += fraction
    return weights


def _adjoint_derivative(
    op0: NonuniformOperator,
    V0: np.ndarray,
    A0,
    free: np.ndarray,
    dA,
    drhs: np.ndarray,
    target: float,
) -> float:
    weights = _interpolation_weights(op0.S, target)
    forcing = drhs[free] - dA[free, :] @ V0[1:-1]
    adjoint = spsolve(A0[free, :][:, free].T.tocsc(), weights[free])
    return float(adjoint @ forcing)


def compute_sensitivities(N: int = 2400) -> list[dict[str, float | str]]:
    """Compute adjoint and direct finite-difference sensitivities."""
    S = uniform_grid(N, S_MAX)
    op0 = _operator_at(S, 'scale', 1.0)
    V0, _, _, _ = solve_operator(op0)
    A0 = interior_matrix(op0)
    slack = V0[1:-1] - op0.psi[1:-1]
    free = np.flatnonzero(slack > TOL_SLACK)

    rows: list[dict[str, float | str]] = []
    for parameter in PARAMETERS:
        base = _parameter_value(parameter)
        eps = 1e-3 if parameter == 'scale' else 1e-5
        op_plus = _operator_at(S, parameter, base + eps)
        op_minus = _operator_at(S, parameter, base - eps)
        dA = (interior_matrix(op_plus) - interior_matrix(op_minus)) / (2.0 * eps)
        drhs = (op_plus.rhs - op_minus.rhs) / (2.0 * eps)
        V_plus, _, _, _ = solve_operator(op_plus)
        V_minus, _, _, _ = solve_operator(op_minus)

        for target in TARGETS:
            adjoint_value = _adjoint_derivative(
                op0, V0, A0, free, dA, drhs, target
            )
            direct_value = (
                float(np.interp(target, S, V_plus))
                - float(np.interp(target, S, V_minus))
            ) / (2.0 * eps)
            scale = max(abs(direct_value), 1e-14)
            rows.append(
                {
                    'parameter': parameter,
                    'target': target,
                    'adjoint': adjoint_value,
                    'direct': direct_value,
                    'abs_error': abs(adjoint_value - direct_value),
                    'rel_error': abs(adjoint_value - direct_value) / scale,
                }
            )
    return rows


def plot_sensitivities(
    rows: list[dict[str, float | str]],
    N: int = 2400,
) -> Path:
    """Plot local-volatility sensitivity profiles and the adjoint check."""
    S = uniform_grid(N, S_MAX)
    grid = np.linspace(70.0, 150.0, 161)
    colors = {'scale': 'tab:blue', 'r': 'tab:red', 'q': 'tab:green'}
    if not rows:
        raise ValueError('sensitivity rows are empty')

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8))
    for parameter in PARAMETERS:
        base = _parameter_value(parameter)
        eps = 1e-3 if parameter == 'scale' else 1e-5
        op_plus = _operator_at(S, parameter, base + eps)
        op_minus = _operator_at(S, parameter, base - eps)
        V_plus, _, _, _ = solve_operator(op_plus)
        V_minus, _, _, _ = solve_operator(op_minus)
        profile = (V_plus - V_minus) / (2.0 * eps)
        profile_axis = axes[0] if parameter == 'scale' else axes[1]
        profile_axis.plot(
            grid,
            np.interp(grid, S, profile),
            color=colors[parameter],
            label=parameter,
        )
        selected = [row for row in rows if row['parameter'] == parameter]
        profile_axis.plot(
            [row['target'] for row in selected],
            [row['adjoint'] for row in selected],
            'o', color=colors[parameter], markersize=4,
        )
        axes[2].semilogy(
            [row['target'] for row in selected],
            [max(float(row['rel_error']), 1e-16) for row in selected],
            'o-', color=colors[parameter], label=parameter,
        )

    axes[0].set_xlabel('Stock price $S$')
    axes[0].set_ylabel(r'$\partial V / \partial \alpha$')
    axes[0].set_title('(a) Volatility-scale sensitivity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel('Stock price $S$')
    axes[1].set_ylabel(r'$\partial V / \partial \theta$')
    axes[1].set_title('(b) Rate and dividend sensitivities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[1].axhline(0.0, color='black', linewidth=0.8)

    axes[2].set_xlabel('Target price $S$')
    axes[2].set_ylabel('Relative adjoint error')
    axes[2].set_title('(c) Adjoint versus direct finite difference')
    axes[2].legend()
    axes[2].grid(True, alpha=0.25, which='both')

    fig.tight_layout()
    output = Path(__file__).resolve().with_name('local_vol_adjoint.png')
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output


def main() -> None:
    rows = compute_sensitivities()
    output = plot_sensitivities(rows)

    print('\nLocal-volatility adjoint sensitivities (N=2400)')
    print(f"{'parameter':>10} {'target':>8} {'adjoint':>14} {'direct FD':>14} {'rel. err.':>12}")
    for row in rows:
        print(
            f"{row['parameter']:>10} {row['target']:8.1f} "
            f"{row['adjoint']:14.6e} {row['direct']:14.6e} "
            f"{row['rel_error']:12.3e}"
        )

    csv_path = Path(__file__).resolve().with_name('local_vol_adjoint.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {csv_path.name} and {output.name}')


if __name__ == '__main__':
    main()
