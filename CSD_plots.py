import numpy as np
import matplotlib.pyplot as plt
import Analysis_module as ana
import Simulation_module as sim
import Algorithm_module as alg

BETAJ_CRIT = 0.4407

betaJ_range = np.concatenate(
    [
        np.linspace(0.2, 0.42, 4),
        [
            BETAJ_CRIT - 0.01,
            BETAJ_CRIT,
            BETAJ_CRIT + 0.01,
        ],  # Finer selection of values near betaJ_crit since metrics vary quickly in this region
        [0.46, 0.54, 0.7],  # Omitting anomalous value at 0.62
    ]
)


def plot_all():
    """
    To illustrate the reduction of CSD by cluster algos
    """
    algorithms = [alg.metropolis_hastings, alg.wolff, alg.swendsen_wang]
    autocorr_data = np.zeros((3, int(len(betaJ_range))))

    for j, algo in enumerate(algorithms):
        for i, betaJ in enumerate(betaJ_range):

            autocorr_series = sim.run_simulation(128, betaJ, algo, 30000)[2]
            tau = ana.find_autocorrelation_time(autocorr_series)

            autocorr_data[j, i] = tau

    for i, algo in enumerate(algorithms):
        taus = autocorr_data[i, :]
        plt.plot(betaJ_range, taus, "o-", label=algo.__name__)

    plt.axvline(BETAJ_CRIT, color="red", linestyle="--", label=r"$\beta_c J$")
    plt.xlabel(r"$\beta J$")
    plt.ylabel(r"Autocorrelation Time $\tau$")
    plt.title("Critical Slowing Down for All Algorithms in 2D Ising Model (L=128)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# def mh_only():
#     autocorr_data = []

#     for betaJ in betaJ_range:

#         autocorr_series = sim.run_simulation(
#             128, betaJ, alg.metropolis_hastings, 30000
#         )[2]
#         tau = ana.find_autocorrelation_time(autocorr_series)
#         autocorr_data.append(tau)

#     plt.plot(betaJ_range, autocorr_data, "o-", label="Metropolis_Hastings")

#     plt.axvline(BETAJ_CRIT, color="red", linestyle="--", label=r"$\beta_c J$")
#     plt.xlabel(r"$\beta J$")
#     plt.ylabel(r"Autocorrelation Time $\tau$")
#     plt.title(
#         "Critical Slowing Down of Metropolis-Hastings Algorithm in 2D Ising Model (L=128)"
#     )
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()


def cluster_plot():
    """
    Plotting only the cluster algos since on much smaller scale than MH
    """
    algorithms = [alg.wolff, alg.swendsen_wang]
    autocorr_data = np.zeros((2, int(len(betaJ_range))))

    for j, algo in enumerate(algorithms):
        for i, betaJ in enumerate(betaJ_range):

            autocorr_series = sim.run_simulation(128, betaJ, algo, 30000)[2]
            tau = ana.find_autocorrelation_time(autocorr_series)

            autocorr_data[j, i] = tau

    for i, algo in enumerate(algorithms):
        taus = autocorr_data[i, :]
        plt.plot(betaJ_range, taus, "o-", label=algo.__name__)

    plt.axvline(BETAJ_CRIT, color="red", linestyle="--", label=r"$\beta_c J$")
    plt.xlabel(r"$\beta J$")
    plt.ylabel(r"Autocorrelation Time $\tau$")
    plt.title("Critical Slowing Down for Cluster Algorithms in 2D Ising Model (L=128)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
