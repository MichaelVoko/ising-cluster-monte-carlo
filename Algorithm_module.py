import numpy as np

rng = np.random.default_rng()


def init_lattice(width, type):
    # Produce an initial lattice with spins. Type 1 gives all spins 1, type 0 gives a random spins. Captures invalid types.

    lattice = np.zeros((width, width))
    if type == 1:
        lattice[:] = 1
    elif type == 0:
        lattice[:] = rng.choice(
            [-1, 1], size=(width, width)
        )  # Generate a random sample of lattice points of +-1

    else:
        raise TypeError("Type is invalid")

    return lattice


def neighbouring_sites(i, j, width):
    # Return the coordinates of the 4 sites adjacent to [i,j] on an width*width lattice.
    # Takes into account periodic boundary conditions.

    # Use modulo operation for periodic boundary conditions to wrap around edges of lattice
    left = ((i - 1) % width, j)
    right = ((i + 1) % width, j)
    up = (i, (j - 1) % width)
    down = (i, (j + 1) % width)

    return [left, right, up, down]


def neighbouring_spins_sum(i, j, lattice, width):
    # Sums the spins of all neighbours of the spin at [i,j].

    spin_sum = 0
    for a in range(0, len(neighbouring_sites(i, j, width))):
        spin_sum += lattice[neighbouring_sites(i, j, width)[a]]

    return spin_sum


def compute_betaDeltaE(i, j, lattice, width, betaJ):
    # Computes the energy difference between the old and new state if spin [i,j] is flipped.

    betaDeltaE = (
        2 * betaJ * lattice[i, j] * neighbouring_spins_sum(i, j, lattice, width)
    )

    return betaDeltaE


def metropolis_hastings(lattice, betaJ):
    # Applies the Metropolis-Hastings algorithm to try and flip a spin.

    # Starting from random point on lattice
    lattice_size = len(lattice)
    i, j = np.random.randint(0, lattice_size, size=2)

    # Calculate change in energy
    dE = compute_betaDeltaE(i, j, lattice, lattice_size, betaJ)
    p_acc = np.exp(-dE)
    accepted_move = 0
    # Apply criterion: probability of one if dE < 0 since lowering energy and random chance of flipping according to the Boltzmann-like probability
    if dE < 0 or np.random.random() < p_acc:
        lattice[(i, j)] = lattice[(i, j)] * -1
        accepted_move += 1

    return accepted_move


def compute_magnetisation(lattice):
    return np.abs(np.sum(lattice) / (len(lattice) ** 2))


"""
Wolff algorithm 
"""


def wolff(lattice, betaJ):
    """
    Perform one Wolff cluster flip on the Ising lattice.
    Returns the updated lattice and the cluster size.
    """
    lattice_size = len(lattice)
    # Selecting random spin
    i0, j0 = np.random.randint(0, lattice_size, size=2)
    spin0 = lattice[i0, j0]

    cluster = {(i0, j0)}
    stack = [(i0, j0)]
    p_add = 1 - np.exp(-2 * betaJ)  # Probability to maintain detailed balance

    while stack:
        i, j = stack.pop()
        for ni, nj in neighbouring_sites(i, j, lattice_size):
            if (ni, nj) not in cluster and lattice[ni, nj] == spin0:
                if np.random.random() < p_add:
                    cluster.add((ni, nj))
                    stack.append((ni, nj))
    # Flip all spins in cluster
    for i, j in cluster:
        lattice[i, j] = -spin0

    return len(cluster)  # Return cluster size for future analysis


"""
Swendsen-Wang algorithm
"""

from collections import deque


def build_clusters(lattice, betaJ):
    """
    Allocating every single site to a cluster to be flipped in the swendsen_wang() function
    """
    lattice_size = len(lattice)
    visited = np.zeros((lattice_size, lattice_size), dtype=bool)
    clusters = []
    p = 1 - np.exp(-2 * betaJ)

    for i in range(lattice_size):
        for j in range(lattice_size):
            if not visited[i, j]:
                cluster = []
                queue = deque([(i, j)])
                visited[i, j] = True

                while queue:
                    x, y = queue.popleft()
                    cluster.append((x, y))
                    for n_i, n_y in neighbouring_sites(x, y, lattice_size):
                        if (
                            not visited[n_i, n_y]  # Not yet visited condition
                            and lattice[n_i, n_y]
                            == lattice[x, y]  # Same spin condition
                            and np.random.rand()
                            < p  # Probability of addition to cluster
                        ):
                            visited[n_i, n_y] = True
                            queue.append((n_i, n_y))  # Next one
                clusters.append(cluster)
    return clusters


def swendsen_wang(lattice, betaJ):
    """
    Each cluster has 50% chance of being flipped
    """
    clusters = build_clusters(lattice, betaJ)
    for cluster in clusters:
        if np.random.rand() < 0.5:
            for i, j in cluster:
                lattice[i, j] *= -1
    avg_cluster_size = np.mean([len(c) for c in clusters])

    return avg_cluster_size  # Return mean cluster size for future analysis


def normalized_step(lattice, betaJ, algorithm):
    """
    Can't compare different algorithms if not normalised by total lattice sites since each step in algos affect a different number of spins
    """

    if algorithm.__name__ == "metropolis_hastings":
        # One full Metropolis sweep
        acceptance_rate = [0] * len(lattice) ** 2
        for i in range(len(lattice) ** 2):

            acceptance_rate[i] = algorithm(lattice, betaJ)

        return np.mean(acceptance_rate)

    elif algorithm.__name__ == "wolff":
        # Perform Wolff sweeps until ~L^2 spins are flipped
        total_flipped = 0
        cluster_sizes = []
        while total_flipped < len(lattice) ** 2:
            flipped = algorithm(lattice, betaJ)
            cluster_sizes.append(flipped)
            total_flipped += flipped

        return np.mean(cluster_sizes)

    elif algorithm.__name__ == "swendsen_wang":
        # One full SW update already touches entire lattice
        return algorithm(lattice, betaJ)

    else:
        raise ValueError("Invalid input")
