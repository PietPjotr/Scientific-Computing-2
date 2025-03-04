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
    gs = GrayScott(100) #, k=0.062, f=0.035, Du=0.16, Dv=0.08)
    nr_frames = 1000
    gs.animate(num_frames=nr_frames, title=f"GrayScott_{nr_frames}frames")
    for i in range(nr_frames):
        gs.Reaction()
        if i % 200 == 0:
            print(f"Iteration {i}")
    
    # Plot the final state
    gs.plot(title=f"GS_pattern_{nr_frames}frames")
    


if __name__ == "__main__":
    # main()
    # eta_evaluations()
    gray_scott()