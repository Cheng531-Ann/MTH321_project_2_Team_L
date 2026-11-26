'''Quantify the positivity obstruction for a correlated nine-point stencil.'''

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stage2d_spread import (
    K,
    L,
    SIGMA1,
    SIGMA2,
    solve_spread_lcp,
)


GRID = 80
RHOS = (0.1, 0.3)


def analyse(rho: float, N: int = GRID) -> dict[str, float]:
    result = solve_spread_lcp(N)
    S = result.S[1:-1]
    V = result.V[1:-1, 1:-1]
    psi = np.maximum(K - S[:, None] + S[None, :], 0.0)
    active = V - psi <= 1e-8

    a = 0.5 * SIGMA1**2 * S[:, None] ** 2
    b = 0.5 * SIGMA2**2 * S[None, :] ** 2
    c = rho * SIGMA1 * SIGMA2 * S[:, None] * S[None, :]
    gamma = 0.5 * np.abs(c)
    margin = np.minimum(a, b) / np.maximum(gamma, 1e-30)
    monotone = margin >= 1.0

    h = L / N
    axis1 = -(a - gamma) / h**2
    axis2 = -(b - gamma) / h**2
    ne = -c / (4.0 * h**2) - gamma / (2.0 * h**2)
    nw = c / (4.0 * h**2) - gamma / (2.0 * h**2)
    se = c / (4.0 * h**2) - gamma / (2.0 * h**2)
    sw = -c / (4.0 * h**2) - gamma / (2.0 * h**2)
    off_diagonal = np.concatenate(
        (
            axis1.ravel(),
            axis2.ravel(),
            ne.ravel(),
            nw.ravel(),
            se.ravel(),
            sw.ravel(),
        )
    )
    positive = off_diagonal > 1e-14

    return {
        'N': N,
        'rho': rho,
        'interior_nodes': int(S.size * S.size),
        'monotone_all_fraction': float(np.mean(monotone)),
        'monotone_free_fraction': float(np.mean(monotone[~active])),
        'monotone_active_fraction': float(np.mean(monotone[active])),
        'min_margin': float(np.min(margin)),
        'max_off_diagonal': float(np.max(off_diagonal)),
        'positive_off_diagonal_fraction': float(np.mean(positive)),
        'free_nodes_with_violation': int(np.sum((~monotone) & (~active))),
    }


def main() -> None:
    rows = [analyse(rho) for rho in RHOS]
    print('Correlated nine-point positivity analysis')
    print(' rho  monotone/all  monotone/free  max off-diag  positive frac')
    for row in rows:
        print(
            '%.1f  %12.4f  %13.4f  %12.3e  %13.4f'
            % (
                row['rho'],
                row['monotone_all_fraction'],
                row['monotone_free_fraction'],
                row['max_off_diagonal'],
                row['positive_off_diagonal_fraction'],
            )
        )

    fields = tuple(rows[0].keys())
    path = Path(__file__).resolve().parent / 'spread2d_nine_point.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted['monotone_all_fraction'] = '%.6e' % row['monotone_all_fraction']
            formatted['monotone_free_fraction'] = '%.6e' % row['monotone_free_fraction']
            formatted['monotone_active_fraction'] = '%.6e' % row['monotone_active_fraction']
            formatted['min_margin'] = '%.6e' % row['min_margin']
            formatted['max_off_diagonal'] = '%.6e' % row['max_off_diagonal']
            formatted['positive_off_diagonal_fraction'] = (
                '%.6e' % row['positive_off_diagonal_fraction']
            )
            writer.writerow(formatted)
    print('Saved %s' % path.name)


if __name__ == '__main__':
    main()
