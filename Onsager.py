import Simulation_module as sim
import numpy as np
import Algorithm_module as alg
import matplotlib.pyplot as plt


# Exact Onsager solution for T < Tc
def exact_ising_magnetisation(betaJs):
    betaJ_c = 0.5 * np.log(1 + np.sqrt(2))
    m = np.zeros_like(betaJs)
    for i, b in enumerate(betaJs):
        if b > betaJ_c:
            sinh_2b = np.sinh(2 * b)
            m[i] = (1 - sinh_2b**-4) ** (1 / 8)
        else:
            m[i] = 0
    return m


def Onsager_plot(betaJ_range):
    # Continuous betaJ values for Onsager solution
    betaJ_vals = np.linspace(0.2, 0.7, 200)
    exact_m_vals = exact_ising_magnetisation(betaJ_vals)

    mags_mh = []
    errors_mh = []

    for betaJ in betaJ_range:

        mag_mh, error_mh, ac, acerr, val = sim.run_simulation(
            128, betaJ, alg.metropolis_hastings, 15000
        )
        mag_mhmean = np.mean(mag_mh[14000:])
        mags_mh.append(mag_mhmean)
        error_mhmean = np.std(mag_mh[14000:]) / np.sqrt(1000)
        errors_mh.append(error_mhmean)

    mags_wolff = []
    errors_wolff = []

    for betaJ in betaJ_range:

        mags_w, error_w, ac, acerr, val = sim.run_simulation(
            128, betaJ, alg.wolff, 5000
        )
        mag_w_mean = np.mean(mags_w[4000:])
        mags_wolff.append(mag_w_mean)
        error_wolff = np.std(mags_w[4000:]) / np.sqrt(1000)
        errors_wolff.append(error_wolff)

    mags_sw = []
    errors_sw = []

    for betaJ in betaJ_range:

        mag_sw, error_sw, ac, acerr, val = sim.run_simulation(
            128, betaJ, alg.swendsen_wang, 5000
        )
        mag_swmean = np.mean(mag_sw[4000:])
        mags_sw.append(mag_swmean)
        error_swmean = np.std(mag_sw[4000:]) / np.sqrt(1000)
        errors_sw.append(error_swmean)

    # Plotting
    # plt.figure(figsize=(8, 6))
    plt.plot(
        betaJ_vals, exact_m_vals, label="Onsager (Exact)", color="white", linestyle="--"
    )

    plt.errorbar(
        betaJ_range,
        mags_mh,
        yerr=errors_mh,
        fmt="o-",
        capsize=4,
        label="Metropolis",
        color="blue",
    )

    plt.errorbar(
        betaJ_range,
        mags_wolff,
        yerr=errors_wolff,
        fmt="s-",
        capsize=4,
        label="Wolff",
        color="green",
    )

    plt.errorbar(
        betaJ_range,
        mags_sw,
        yerr=errors_sw,
        fmt="^-",
        capsize=4,
        label="Swendsen-Wang",
        color="red",
    )

    plt.xlabel(r"$\beta J$")
    plt.ylabel("Magnetisation")
    plt.title("Magnetisation vs beta J (Simulations vs Onsager)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
