"""Plot the policy-iteration versus PSOR benchmark from the experiment."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from experiment_solvers import SOLVER_CSV, benchmark, write_solver_comparison


def load_benchmark() -> dict[str, float | int]:
    """Load the same benchmark record that supplies the report table."""
    if not SOLVER_CSV.exists():
        data = benchmark()
        write_solver_comparison(data)

    with SOLVER_CSV.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    return {
        "N": int(row["N"]),
        "policy_iterations": int(row["policy_iterations"]),
        "policy_time": float(row["policy_time"]),
        "psor_iterations": int(row["psor_iterations"]),
        "psor_time": float(row["psor_time"]),
    }


def main() -> None:
    data = load_benchmark()
    solvers = ("Policy Iteration", "PSOR")
    iterations = np.array(
        [data["policy_iterations"], data["psor_iterations"]],
        dtype=float,
    )
    cpu_times = np.array(
        [data["policy_time"], data["psor_time"]],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    axes[0].bar(solvers, iterations, color=("tab:blue", "tab:orange"))
    axes[0].set_yscale("log")
    axes[0].set_title("Iterations")
    axes[0].set_ylabel("Iterations (log scale)")
    for index, value in enumerate(iterations):
        axes[0].text(
            index,
            value * 1.12,
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )

    axes[1].bar(solvers, cpu_times, color=("tab:blue", "tab:orange"))
    axes[1].set_yscale("log")
    axes[1].set_title("CPU Time")
    axes[1].set_ylabel("CPU time in seconds (log scale)")
    for index, value in enumerate(cpu_times):
        axes[1].text(
            index,
            value * 1.12,
            f"{value:.3f}s",
            ha="center",
            va="bottom",
        )

    for axis in axes:
        axis.set_ylim(bottom=1e-2)
        axis.grid(axis="y", which="both", linestyle=":", alpha=0.45)

    fig.suptitle(f"Solver Comparison (N={data['N']})")
    fig.tight_layout()
    print("Saved solver_comparison.png")
    print(
        f"Policy Iteration: {iterations[0]:.0f} iterations, "
        f"{cpu_times[0]:.3f} s"
    )
    print(f"PSOR:            {iterations[1]:.0f} iterations, {cpu_times[1]:.3f} s")
    plt.savefig("solver_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
