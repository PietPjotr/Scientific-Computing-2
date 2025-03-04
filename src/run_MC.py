"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Runs the monte carlo (MC) class for different parameter settings
and stores them inside the data folder for analysis/plotting. This file uses
very primitive parallization by splitting the 4 different parameter settings
over 4 different processes. This is not quite ideal since one parameter setting
can take significantly longer than a different one, but atleast this helps with
some speed up and it's very easy to implement.

Typical usage example:

python3 run_MC.py
"""

import multiprocessing as mp
from MC import MC
from datetime import datetime


def run_mc(max_iter, ps, timestamp):
    print(f"Running MC with ps={ps}")
    mc = MC(100, ps=ps, max_iter=max_iter, timestamp=timestamp)
    mc.run()
    mc.save_to_csv()


def main():
    max_iter = 1000
    ps_values = [0.04, 0.03, 0.02, 0.01]

    processes = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ps in ps_values:
        p = mp.Process(target=run_mc, args=(max_iter, ps, timestamp))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All simulations completed and saved.")


if __name__ == "__main__":
    main()
