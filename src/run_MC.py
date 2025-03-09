"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Runs the monte carlo (MC) class for different parameter settings
and stores them inside the data folder for analysis/plotting. The largest
values run for our experiment takes around 4 hours so I would not recomment
running this again. Therefore the main also contains a much smaller version
that will take ~5 minutes.

Typical usage example:

python3 run_MC.py
"""

from MC import MC
from datetime import datetime


def run_mc_sequential(max_iter=1000, ps_values=[1, 0.75, 0.5, 0.25]):
    """
    This function runs len(ps_values) amount of MC class objects, and stores
    the results in the designated folder ../data/MC/timestamp,

    Args:
        max_iter: the amount of max iterations per run (int)
        ps_values: list containing the requested ps values [float]
    """
    assert isinstance(ps_values, list), "ps_values must be a list"
    assert all(0 <= p <= 1 for p in ps_values), \
           "All elements in ps_values must be floats between 0 and 1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ps in ps_values:
        print(f"Running MC with ps={ps}")
        mc = MC(100, ps=ps, max_iter=max_iter, timestamp=timestamp)
        try:
            mc.run()
        except StopIteration:
            print(f"Simulation stopped early for ps={ps} due to iteration limit" +
                   " or reaching the top of the grid.")
        mc.save_to_csv()

    print("All simulations completed and saved.")


if __name__ == "__main__":
    run_mc_sequential()
