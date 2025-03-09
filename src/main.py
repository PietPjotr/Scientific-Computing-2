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
from DLA import DLA
from GrayScott import *
from plot_MC import mc_report_plot, mc_plot_data
from run_MC import run_mc_sequential
from plot_GrayScott import plot_results
from dla_analysis import analyze_eta_influence, visualize_results

import os

def run_simple_dla():
    """Run a simple DLA demonstration"""
    print("Running simple DLA demonstration")
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
    gs = GrayScott(100) #, k=0.062, f=0.035, Du=0.16, Dv=0.08)
    nr_frames = 1000
    gs.animate(num_frames=nr_frames, title=f"GrayScott_{nr_frames}frames")
    for i in range(nr_frames):
        gs.Reaction()
        if i % 200 == 0:
            print(f"Iteration {i}")

    # Plot the final state
    gs.plot(title=f"GS_pattern_{nr_frames}frames")


def run_eta_evaluations():
    """Run DLA simulations with different eta values and create animations"""
    print("Running DLA with multiple eta values")

    os.makedirs("eta_figures", exist_ok=True)

    # for eta in [1, 3, 5, 8, 10]:
    #     print(f"Processing eta = {eta}")
    #     dla = DLA(100, eta=eta)
    #     dla.animate(num_frames=1000, title=f"eta_figures/DLA_eta{eta}")
    #     dla.plot(title=f"eta_figures/DLA_eta{eta}")
    visualize_results()
    

def run_gray_scott():
    """Run Gray-Scott simulations with different parameter values"""
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

    # Mitosis simulation
    # gs = GrayScott(100, k=0.06, f=0.025, Du=0.2, Dv=0.08, noise=True)
    # gs.animate(num_frames=2000, title="GrayScott_mitosis_0.2")


def main():
    """Main function to run the selected simulation"""
    # Uncomment the function you want to run

    # Simple DLA demonstration
    # run_simple_dla()

    # DLA eta parameter analysis (run simulations)
    #analyze_eta_influence()
    
    # Visualize previously saved DLA results
    visualize_results()
    
    # Gray-Scott simulations
    #run_gray_scott()

    # MC run:
    # run_mc_parallel(1000)
    # mc_plot_data()
    # mc_plot_data()

    # MC plotting: Plots the already computed data for MC (question b)
    # mc_report_plot
    # MC plotting:
    # report_plot()


if __name__ == "__main__":
    main()
