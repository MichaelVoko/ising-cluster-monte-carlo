import Algorithm_module as alg
import matplotlib.pyplot as plt
import numpy as np
import Simulation_module as sim

BETAJ_CRIT = 0.4407


def plot_histogram(sweeps, lattice_size, algo, betaJ, temp, width):
    name = algo.__name__
    cluster_size = [0] * sweeps
    lattice = alg.init_lattice(lattice_size, 0)

    for i in range(sweeps):
        length = algo(lattice, betaJ)
        cluster_size[i] = length

    cluster_sizes = cluster_size[
        int(sweeps * 0.2) :
    ]  # Discarding initial results to produce data more representative of eqm

    plt.hist(
        cluster_sizes, bins=np.arange(1, max(cluster_sizes) + 1, width), density=True
    )

    plt.xlabel("Cluster Size")
    plt.ylabel("Relative Frequency")
    plt.title(f"Cluster Size Distribution for {name} Algorithm at {temp} Temperature")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    print("Average cluster size:", np.mean(cluster_sizes))


def cluster_plot(betaJ_range, lattice_size):
    Wolff_cluster_sizes = [0] * len(betaJ_range)
    Wolff_size_errors = [0] * len(betaJ_range)
    SW_cluster_sizes = [0] * len(betaJ_range)
    SW_size_errors = [0] * len(betaJ_range)

    for i, betaJ in enumerate(betaJ_range):
        values = sim.run_simulation(lattice_size, betaJ, alg.wolff, 10000)[4]
        cluster_size = values[2]
        size_error = values[3]
        Wolff_cluster_sizes[i] = cluster_size
        Wolff_size_errors[i] = size_error

    for i, betaJ in enumerate(betaJ_range):
        values = sim.run_simulation(lattice_size, betaJ, alg.swendsen_wang, 10000)[4]
        cluster_size = values[2]
        size_error = values[3]
        SW_cluster_sizes[i] = cluster_size
        SW_size_errors[i] = size_error

    plt.figure()
    plt.errorbar(betaJ_range, Wolff_cluster_sizes, yerr=Wolff_size_errors, marker="o")
    plt.xlabel(r"$\beta J$")
    plt.ylabel("Mean cluster size")
    plt.title(f"Wolff vs $\\beta J$ (L={lattice_size})")
    plt.grid(True)
    plt.axvline(x=BETAJ_CRIT, color="red", linestyle="--", label=r"$\beta J_c$")
    plt.legend()
    plt.show()

    plt.figure()
    plt.errorbar(betaJ_range, SW_cluster_sizes, yerr=SW_size_errors, marker="o")
    plt.xlabel(r"$\beta J$")
    plt.ylabel("Acceptance Rate")
    plt.title(f"Wolff vs $\\beta J$ (L={lattice_size})")
    plt.grid(True)
    plt.axvline(x=BETAJ_CRIT, color="red", linestyle="--", label=r"$\beta J_c$")
    plt.legend()
    plt.show()


def accept_plot(betaJ_range, lattice_size):
    acceptance_rates = [0] * len(betaJ_range)
    acceptance_errors = [0] * len(betaJ_range)

    for i, betaJ in enumerate(betaJ_range):
        values = sim.run_simulation(
            lattice_size, betaJ, alg.metropolis_hastings, 10000
        )[4]
        acc_rate = values[2]
        acc_err = values[3]
        acceptance_rates[i] = acc_rate
        acceptance_rates[i] = acc_err

    plt.errorbar(betaJ_range, acceptance_rates, yerr=acceptance_errors, marker="o")
    plt.xlabel(r"$\beta J$")
    plt.ylabel("Acceptance Rate")
    plt.title(f"Metropolis-Hastings Acceptance Rate vs Temperature (L={lattice_size})")
    plt.grid(True)
    plt.axvline(x=BETAJ_CRIT, color="red", linestyle="--", label=r"$\beta_{crit} J$")
    plt.legend()
    plt.show()
