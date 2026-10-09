import importlib
import Simulation_module as sim
import Analysis_module as ana

importlib.reload(ana)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import Algorithm_module as alg
import numpy as np


BETAJ_CRIT = 0.4407


def power_law(L, A, z):
    return A * L**z


def power_plot(lattice_sizes, algo):
    """
    Determining dynamic exponents by fitting ac times with different lattice sizes to power law
    """
    autocorr_times = []

    for L in lattice_sizes:
        print(f"Running simulation for L = {L}")
        autocorr, autoerr = sim.run_simulation(L, BETAJ_CRIT, algo, 15000)[2:4]
        tau = ana.find_autocorrelation_time(autocorr, True)
        autocorr_times.append(tau)

    L_array = np.array(lattice_sizes)
    tau_array = np.array(autocorr_times)

    popt, _ = curve_fit(power_law, L_array, tau_array)
    A_fit, z_fit = popt

    plt.figure(figsize=(7, 5))
    plt.loglog(L_array, tau_array, "o", label="Data")
    plt.loglog(
        L_array, power_law(L_array, *popt), "--", label=f"Fit: $z$ = {z_fit:.2f}"
    )
    plt.xlabel("Lattice size $L$")
    plt.ylabel("Autocorrelation time $\\tau$")
    plt.title(f"Scaling of Autocorrelation Time at $T_c$ for {algo.__name__} Algorithm")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
