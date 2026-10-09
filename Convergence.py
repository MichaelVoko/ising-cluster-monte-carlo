import numpy as np
import matplotlib.pyplot as plt
import Simulation_module as sim
import Algorithm_module as alg


def convergence_simulation(lattice_size, betaJ, algorithm, sweeps):
    num_simulations = 5

    for sim_number in range(num_simulations):
        ordered_lattice = alg.init_lattice(lattice_size, 1)
        disordered_lattice = alg.init_lattice(lattice_size, 0)

        ordered_magnetisations = [0] * sweeps
        disordered_magnetisations = [0] * sweeps

        for current in range(sweeps):
            algorithm(ordered_lattice, betaJ)
            O_mag = alg.compute_magnetisation(ordered_lattice)
            ordered_magnetisations[current] = O_mag

            algorithm(disordered_lattice, betaJ)
            D_mag = alg.compute_magnetisation(disordered_lattice)
            disordered_magnetisations[current] = D_mag

        if sim_number == 0:
            all_Omagnetisations = np.asarray(ordered_magnetisations)
            all_Dmagnetisations = np.asarray(disordered_magnetisations)

        else:

            all_Omagnetisations = np.vstack(
                [all_Omagnetisations, np.asarray(ordered_magnetisations)]
            )
            all_Dmagnetisations = np.vstack(
                [all_Dmagnetisations, np.asarray(disordered_magnetisations)]
            )

    disordered_magnetisations = np.mean(all_Dmagnetisations, axis=0)
    disordered_errors = np.std(all_Dmagnetisations, axis=0) / np.sqrt(num_simulations)

    ordered_magnetisations = np.mean(all_Omagnetisations, axis=0)
    ordered_errors = np.std(all_Omagnetisations, axis=0) / np.sqrt(num_simulations)

    return (
        disordered_magnetisations,
        disordered_errors,
        ordered_magnetisations,
        ordered_errors,
    )


def plot_convergence(algo, sweeps, lattice_size, betaJ):
    disordered, d_errors, ordered, o_errors = convergence_simulation(
        lattice_size, betaJ, algo, sweeps
    )

    # Make sure all arrays are the correct length
    ordered = np.array(ordered)
    o_errors = np.array(o_errors)
    disordered = np.array(disordered)
    d_errors = np.array(d_errors)

    sweep_range_o = np.arange(len(ordered))
    sweep_range_d = np.arange(len(disordered))

    plt.plot(sweep_range_o, ordered, label="Ordered start")
    plt.fill_between(
        sweep_range_o,
        ordered - o_errors,
        ordered + o_errors,
        alpha=0.1,
        label="Ordered error",
    )

    plt.plot(sweep_range_d, disordered, label="Disordered start")
    plt.fill_between(
        sweep_range_d,
        disordered - d_errors,
        disordered + d_errors,
        alpha=0.1,
        label="Disordered error",
    )

    plt.xlabel("Sweep Number")
    plt.ylabel("Average Magnetisation")
    plt.ylim(0, 1)
    plt.title(
        f"Convergence of Magnetisation ({algo.__name__}, βJ = {betaJ}, N = {lattice_size})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
