'''Plot the signed value-error budget at the production local-volatility mesh.'''

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parent
    with (root / 'stage2_convergence.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    row = next(item for item in rows if int(float(item['N'])) == 2400)
    spatial = float(row['uniform_error_finite_reference'])
    truncation = -float(row['truncation_error_V100'])
    total = spatial + truncation

    labels = [
        'Spatial\ndiscretisation',
        'Far-field\ntruncation',
        'Combined\nproduction error',
    ]
    values = [spatial, truncation, total]
    colors = ['#2a9d8f', '#e76f51', '#264653']

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.62,
        edgecolor='black',
        linewidth=0.6,
    )
    ax.axhline(0.0, color='black', linewidth=0.9)
    ax.set_ylabel(r'Signed error in $V(100)$')
    ax.set_title(r'Error budget at $N=2400$, $h=1$ (local volatility)')
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.0065, 0.0019)

    for bar, value in zip(bars, values):
        offset = 1.5e-4 if value >= 0.0 else -1.5e-4
        va = 'bottom' if value >= 0.0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f'{value:+.3e}',
            ha='center',
            va=va,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(root / 'error_budget.png', dpi=220)
    plt.close(fig)

    with (root / 'error_budget.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('N', 'component', 'signed_error_V100'))
        writer.writerow((2400, 'spatial', f'{spatial:.10e}'))
        writer.writerow((2400, 'far_field_truncation', f'{truncation:.10e}'))
        writer.writerow((2400, 'combined', f'{total:.10e}'))

    print(
        'Saved error_budget.png and error_budget.csv '
        f'(spatial={spatial:.6e}, truncation={truncation:.6e}, total={total:.6e})'
    )


if __name__ == '__main__':
    main()
