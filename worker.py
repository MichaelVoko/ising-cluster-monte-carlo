import Simulation_module as sim
import Algorithm_module as alg
import logging

# Set up logging in case errors occur while running
logging.basicConfig(level=logging.INFO)

fixed_lattice_size = 128
BETAJ_CRIT = 0.4407  # To 4 d.p.


def run_metropolis(betaJ):
    try:
        logging.info(f"Running Metropolis with betaJ={betaJ}")
        return sim.run_simulation(
            fixed_lattice_size, betaJ, alg.metropolis_hastings, 15000
        )
    except Exception as e:
        logging.error(f"Error running Metropolis with betaJ={betaJ}: {e}")
        raise


def run_wolff(betaJ):
    try:
        logging.info(f"Running Wolff with betaJ={betaJ}")
        return sim.run_simulation(
            fixed_lattice_size, betaJ, alg.wolff, 5000
        )  # Fewer sweeps for cluster algos since faster
    except Exception as e:
        logging.error(f"Error running Wolff with betaJ={betaJ}: {e}")
        raise


def run_swendsen(betaJ):
    try:
        logging.info(f"Running Swendsen-Wang with betaJ={betaJ}")
        return sim.run_simulation(fixed_lattice_size, betaJ, alg.swendsen_wang, 5000)
    except Exception as e:
        logging.error(f"Error running Swendsen-Wang with betaJ={betaJ}: {e}")
        raise


def run_metropolis_close(betaJ):
    try:
        logging.info(f"Running Metropolis with betaJ={betaJ} (close to critical)")
        return sim.run_simulation(
            fixed_lattice_size,
            betaJ,
            alg.metropolis_hastings,
            50000,  # More sweeps near criticality to ensure good data
        )
    except Exception as e:
        logging.error(
            f"Error running Metropolis with betaJ={betaJ} (close to critical): {e}"
        )
        raise


def run_wolff_close(betaJ):
    try:
        logging.info(f"Running Wolff with betaJ={betaJ} (close to critical)")
        return sim.run_simulation(fixed_lattice_size, betaJ, alg.wolff, 20000)
    except Exception as e:
        logging.error(
            f"Error running Wolff with betaJ={betaJ} (close to critical): {e}"
        )
        raise


def run_swendsen_close(betaJ):
    try:
        logging.info(f"Running Swendsen-Wang with betaJ={betaJ} (close to critical)")
        return sim.run_simulation(fixed_lattice_size, betaJ, alg.swendsen_wang, 20000)
    except Exception as e:
        logging.error(
            f"Error running Swendsen-Wang with betaJ={betaJ} (close to critical): {e}"
        )
        raise


def run_metropolis_crit(size):
    try:
        logging.info(f"Running Metropolis at critical temperature with size={size}")
        return sim.run_simulation(size, BETAJ_CRIT, alg.metropolis_hastings, 50000)
    except Exception as e:
        logging.error(
            f"Error running Metropolis at critical temperature with size={size}: {e}"
        )
        raise


def run_wolff_crit(size):
    try:
        logging.info(f"Running Wolff at critical temperature with size={size}")
        return sim.run_simulation(size, BETAJ_CRIT, alg.wolff, 30000)
    except Exception as e:
        logging.error(
            f"Error running Wolff at critical temperature with size={size}: {e}"
        )
        raise


def run_swendsen_crit(size):
    try:
        logging.info(f"Running Swendsen-Wang at critical temperature with size={size}")
        return sim.run_simulation(size, BETAJ_CRIT, alg.swendsen_wang, 30000)
    except Exception as e:
        logging.error(
            f"Error running Swendsen-Wang at critical temperature with size={size}: {e}"
        )
        raise
