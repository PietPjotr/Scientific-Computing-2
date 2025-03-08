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
    k_values = 0.117#[0.001, 0.005, 0.01, 0.049] #, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    titles = []
    for k in k_values:	
        gs = GrayScott(100, k=k, f=0.026, Du=0.16, Dv=0.08, noise=True)
        nr_frames = 300
        titles.append(f"k={k} ({nr_frames} frames).csv")
        #gs.animate(num_frames=nr_frames, title=f"GrayScott_{nr_frames}framesK{k}")
        for i in range(nr_frames):
            gs.Reaction()
            if i == nr_frames-1:
                gs.save_to_csv(f"k={k} ({nr_frames} frames)")
    plot_results(titles, title="GrayScott_results")
    # f_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # for f in f_values:	
    #     gs = GrayScott(100, k=0.06, f=f, Du=0.16, Dv=0.08)
    #     nr_frames = 4000
    #     #gs.animate(num_frames=nr_frames, title=f"GrayScott_{nr_frames}frames")
    #     for i in range(nr_frames):
    #         gs.Reaction()
    #         if i % 200 == 0:
    #             print(f"Iteration {i}")
        
    #     # Plot the final state
    #     gs.plot(title=f"GS_pattern_{nr_frames}framesk{f}")
    


if __name__ == "__main__":
    # main()
    # eta_evaluations()
    gray_scott()