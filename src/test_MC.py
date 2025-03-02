"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Combines all the different modules into one and runs the required
parts and plots.

Typical usage example:

python3 test_MC.py
"""

from DLA import *
from MC import *


def main():
    mc = MC(100)
    mc.animate(num_frames=10000)


if __name__ == "__main__":
    main()
