"""Plot Delta and Gamma for the Stage 2 local-volatility solution."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stage2_local_volatility import solve_local_volatility_put


GRID_SIZES = (600, 1200, 2400, 4800)
GAMMA_CSV = Path(__file__).with_name("gamma_peak.csv")


def compute_profile(N: int):
    result = solve_local_volatility_put(N=N)
    S = result.S
    V = result.V
    delta = np.gradient(V, S)
    gamma = np.gradient(delta, S)
    delta_at_100 = float(np.interp(100.0, S, delta))
    max_gamma = float(np.max(gamma))
    return result, delta, gamma, delta_at_100, max_gamma


def write_gamma_refinement(rows: list[dict[str, float | int]]) -> None:
    fields = (
        "N",
        "max_gamma",
        "delta_at_100",
        "S_lo",
        "S_hi",
        "S_free",
    )
    with GAMMA_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, float | int]] = []
    selected = None
    for N in GRID_SIZES:
        result, delta, gamma, delta_at_100, max_gamma = compute_profile(N)
        rows.append(
            {
                "N": N,
                "max_gamma": f"{max_gamma:.9e}",
                "delta_at_100": f"{delta_at_100:.9e}",
                "S_lo": f"{result.S_lo:.9e}",
                "S_hi": f"{result.S_hi:.9e}",
                "S_free": f"{result.S_free:.9e}",
            }
        )
        if N == 2400:
            selected = (result, delta, gamma, delta_at_100, max_gamma)
        print(f"N={N}: max Gamma={max_gamma:.6e}, Delta(100)={delta_at_100:.6f}")

    write_gamma_refinement(rows)
    print(f"Wrote {GAMMA_CSV.name}")

    if selected is None:
        raise RuntimeError("the N=2400 profile required for the figure is missing")
    result, delta, gamma, delta_at_100, max_gamma = selected
    S = result.S
    V = result.V
    S_star = result.S_free

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(S, delta, color="blue", linewidth=1.8, label=r"$\Delta$")
    axes[0].axhline(-1.0, color="black", linestyle=":", linewidth=1.1)
    axes[0].axvline(
        S_star,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$S_h^*={S_star:.2f}$",
    )
    axes[0].axvspan(
        result.S_lo,
        result.S_hi,
        color="red",
        alpha=0.12,
        label=rf"bracket $[{result.S_lo:.2f}, {result.S_hi:.2f}]$",
    )
    axes[0].set_xlim(50.0, 100.0)
    axes[0].set_xlabel("Stock price S")
    axes[0].set_ylabel(r"$\Delta=\partial V/\partial S$")
    axes[0].set_title("Delta")
    axes[0].legend(loc="lower right")

    axes[1].plot(S, gamma, color="darkgreen", linewidth=1.8, label=r"$\Gamma$")
    axes[1].axvline(
        S_star,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$S_h^*={S_star:.2f}$",
    )
    axes[1].axvspan(
        result.S_lo,
        result.S_hi,
        color="red",
        alpha=0.12,
        label=rf"bracket $[{result.S_lo:.2f}, {result.S_hi:.2f}]$",
    )
    axes[1].set_xlim(50.0, 100.0)
    axes[1].set_xlabel("Stock price S")
    axes[1].set_ylabel(r"$\Gamma=\partial^2 V/\partial S^2$")
    axes[1].set_title("Gamma")
    axes[1].legend(loc="upper right")

    fig.suptitle("Stage 2 Local-Volatility Greeks (N=2400)", fontsize=13)
    fig.tight_layout()
    print(f"Figure profile: Delta(100) = {delta_at_100:.6f}")
    print(f"Figure profile: max Gamma = {max_gamma:.6e}")
    print("Saved greeks.png")
    plt.savefig(Path(__file__).with_name("greeks.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
