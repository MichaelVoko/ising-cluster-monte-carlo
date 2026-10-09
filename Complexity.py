import Algorithm_module as alg
import matplotlib.pyplot as plt
import numpy as np
import Simulation_module as sim
from scipy.optimize import curve_fit

BETAJ_CRIT = 0.4407


def plot_complexity(lattice_sizes, algo):
    times = []
    errors = []

    for size in lattice_sizes:
        values = sim.run_simulation(size, BETAJ_CRIT, algo, 50000)[4]
        times.append(values[0])
        errors.append(values[1])

    def power_law(lattice_sizes, a, p):
        return a * lattice_sizes**p

    popt_m, pcov_m = curve_fit(
        power_law, lattice_sizes, times, sigma=errors, absolute_sigma=True
    )
    a_m, p_m = popt_m
    p_err_m = np.sqrt(np.diag(pcov_m))[1]

    plt.figure()
    plt.errorbar(
        lattice_sizes,
        times,
        yerr=errors,
        fmt="o-",
        capsize=4,
        label=f"{algo.__name__} (p ≈ {p_m:.2f} ± {p_err_m:.2f})",
    )
    plt.plot(lattice_sizes, power_law(lattice_sizes, *popt_m), "k--", alpha=0.6)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lattice size (L)")
    plt.ylabel("Time per sweep (seconds))")
    plt.title(f"Scaling of Time per Sweep vs. Lattice Size for {algo.__name__}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Fitted exponent ({algo.__name__}): p = {p_m:.3f} ± {p_err_m:.3f}")
