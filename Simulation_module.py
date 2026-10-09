"""
Simulations
"""

import Algorithm_module as alg
import os
import numpy as np
import time
import Analysis_module as ana


def initial_simulation(lattice_size, betaJ, algorithm, sweeps):
    num_simulations = 5

    times = []

    for sim_number in range(num_simulations):
        lattice = alg.init_lattice(lattice_size, 0)

        magnetisations = [0] * sweeps
        rate_or_size = [0] * sweeps

        start_time = time.time()  # Start timer

        for i in range(sweeps):
            accept_or_length = alg.normalized_step(lattice, betaJ, algorithm)
            mag = alg.compute_magnetisation(lattice)
            magnetisations[i] = mag
            rate_or_size[i] = accept_or_length

        end_time = time.time()

        # (Total time for the simulation)/2 since two lattices used
        total_time = end_time - start_time
        time_per_step = total_time / sweeps  # Time per step
        times.append(time_per_step)  # Append to the list of times

        #
        new_magnetisations = np.array(magnetisations)
        if sim_number == 0:
            all_magnetisations = np.array(new_magnetisations)
        else:
            all_magnetisations = np.vstack([all_magnetisations, new_magnetisations])
        sim_number += 1

    time_per_step_mean, time_per_step_error = ana.find_time_per_step(times, sweeps)

    rate_or_size_mean, rate_or_size_error = ana.find_rate_or_size(
        rate_or_size[int(sweeps * 0.2) :]
    )

    values = [
        time_per_step_mean,
        time_per_step_error,
        rate_or_size_mean,
        rate_or_size_error,
    ]

    magnetisations = np.mean(all_magnetisations, axis=0)
    mag_errors = np.std(all_magnetisations, axis=0) / np.sqrt(num_simulations)

    eqm_magnetisations = all_magnetisations[:, int(sweeps * 0.2) :]

    all_autocorrelations = [
        ana.sample_autocovariance(
            eqm_magnetisations[run, :], int(len(eqm_magnetisations[run, :]) / 2)
        )
        for run in range(num_simulations)
    ]
    autocorrelations = np.mean(all_autocorrelations, axis=0)
    autocorr_errors = np.std(all_autocorrelations, axis=0) / np.sqrt(num_simulations)

    return magnetisations, mag_errors, autocorrelations, autocorr_errors, values


def cache_filename(Storage_type, lattice_size, betaJ, algorithm):
    name = algorithm.__name__
    return f"cache/{name}_N={lattice_size}_J={betaJ:.2f}_{Storage_type}.npz"


def run_simulation(lattice_size, betaJ, algorithm, steps, force_recompute=False):
    magnetisations_file = cache_filename("MV", lattice_size, betaJ, algorithm)
    os.makedirs(os.path.dirname(magnetisations_file), exist_ok=True)

    magerrors_file = cache_filename("ME", lattice_size, betaJ, algorithm)
    os.makedirs(os.path.dirname(magerrors_file), exist_ok=True)

    autocorrelations_file = cache_filename("AV", lattice_size, betaJ, algorithm)
    os.makedirs(os.path.dirname(autocorrelations_file), exist_ok=True)

    autoerrors_file = cache_filename("AE", lattice_size, betaJ, algorithm)
    os.makedirs(os.path.dirname(autoerrors_file), exist_ok=True)

    value_file = cache_filename("V", lattice_size, betaJ, algorithm)
    os.makedirs(os.path.dirname(value_file), exist_ok=True)

    if not force_recompute and os.path.exists(magnetisations_file):
        with np.load(magnetisations_file) as f:
            magnetisations = f["magnetisations"]

        with np.load(magerrors_file) as f:
            mag_errors = f["mag_errors"]

        with np.load(autocorrelations_file) as f:
            autocorrelations = f["autocorrelations"]

        with np.load(autoerrors_file) as f:
            auto_errors = f["auto_errors"]

        with np.load(value_file) as f:
            values = f["values"]

        return magnetisations, mag_errors, autocorrelations, auto_errors, values
    else:

        magnetisations, mag_errors, autocorrelations, auto_errors, values = (
            initial_simulation(lattice_size, betaJ, algorithm, steps)
        )
        np.savez(magnetisations_file, magnetisations=magnetisations)
        np.savez(magerrors_file, mag_errors=mag_errors)
        np.savez(autocorrelations_file, autocorrelations=autocorrelations)
        np.savez(autoerrors_file, auto_errors=auto_errors)
        np.savez(value_file, values=values)

    return magnetisations, mag_errors, autocorrelations, auto_errors, values
