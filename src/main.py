"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Combines all the different modules into one and runs the required
parts and plots.

Typical usage example:

python3 main.py
"""

from DLA import *
from GrayScott import *
from plot_GrayScott import plot_results

def main():
    dla = DLA(100, eta=1)
    dla.animate(num_frames=20)
    # for i in range(10):
    #     dla.plot()
    #     dla.step()

def eta_evaluations():
    for eta in [10]:
        dla = DLA(100, eta=eta)
        dla.animate(num_frames=1000, title=f"eta_figures/DLA_eta{eta}")
        #dla.plot(title=f"eta_figures/DLA_eta{eta}")

def gray_scott():
    f_values =[0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    du_values = [0.12, 0.16, 0.2]
    for du in du_values:
        titles = []
        iterations = []
        for f in f_values:	
            gs = GrayScott(100, k=0.06, f=f, Du=du, Dv=0.08, noise=True)
            nr_iterations = gs.run_until_criterium(threshold=0.1, max_steps=20000) # runs simulation until all concentrations are below the threshold or max_steps is reached
            iterations.append(nr_iterations)
            titles.append(f"f={f} ({nr_iterations} iterations)du{du}.csv")
            gs.save_to_csv(f"f={f} ({nr_iterations} iterations)du{du}")
        plot_results(titles, "f", f_values, iterations, title=f"GrayScott_results_fvaluesdu{du}")

if __name__ == "__main__":
    # main()
    # eta_evaluations()
    gray_scott()