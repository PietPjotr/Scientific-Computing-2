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

def main():
    dla = DLA(100, eta=1)
    dla.animate(num_frames=20)
    # for i in range(10):
    #     dla.plot()
    #     dla.step()

def eta_evaluations():
    for eta in [2, 5, 10]:
        dla = DLA(100, eta=eta)
        dla.animate(num_frames=1000, title=f"eta_figures/DLA_eta{eta}")
        #dla.plot(title=f"eta_figures/DLA_eta{eta}")

if __name__ == "__main__":
    #main()
    eta_evaluations()