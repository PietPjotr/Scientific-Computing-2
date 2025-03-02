"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

foo = ClassFoo()
bar = foo.function_bar()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve      # used for fast double for loop
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime

# global vars indicated by all caps
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)

LABELSIZE = 20
TICKSIZE = 16

class DLA:
    # TODO: add maybe the starting position for the cluster,
    # and or also a way to determine the source/sinc positions
    # standard is top row source, bottom row sink and of course the clusters are sinks as well
    def __init__(self, N, eta=1, tol=1e-4, max_iter=10000, omega=1.9):
        self.eta = eta
        self.N = N
        self.omega = omega
        self.tol = tol

        # initialize the concentration fields and the bitmaps
        self.c = np.zeros((N, N))
        self.sinks = np.zeros((N, N))
        self.sources = np.zeros((N, N))
        self.cluster = np.zeros((N, N))

        self.sources[self.N-1, :] = 1       # top boundary source
        self.sinks[0, :] = 1                # bottom boundary sink
        self.cluster[0, N // 2] = 1    # starting point cluster in the middle

        # initialise candidate array:
        self.candidates = self.init_candidates()

        # update all positions that are sources to 1 and all positions that are
        # sinks to 0 (numpy array indexing magic)
        self.c[self.sources == 1] = 1
        self.c[self.sinks == 1] = 0
        self.c[self.cluster == 1] = 0

        self.NO_steps = 0

    def init_candidates(self):
        """
        Initialises the candidates set.

        This set is used to determine which candidate positions can be added
        to the cluster. The set is used since only a small portion of the grid
        will be candidates and we also want quick access to all candidates.

        Returns:
            candidates: set of position tuples (row, col) that represent the
            current candidates
        """
        candidates = set()

        # neighbour ordering: up right down left
        mask = self.get_mask()
        for r in range(self.N):
            for c in range(self.N):
                if self.cluster[r, c] == 1:
                    for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        nr = r + dr
                        nc = (c + dc)%(self.N-1) # Make sure neighbour is within grid
                        # TODO: ADD ASSERT OR BREAK IF NOT THE CASE
                        if mask[nr, nc] != 1:
                            candidates.add((nr, nc))

        return candidates

    def get_mask(self):
        """
        Gets the bitmask for all constant concentration positions

        Returns:
            mask: 2d numpy array representing all the constant/skippable
            positions
        """
        return (self.sinks.astype(bool) | self.sources.astype(bool) |
                self.cluster.astype(bool))

    def step_SOR(self):
        """
        Perform one iteration of SOR (self organised relaxation).

        Returns:
            delta
        """
        c_old = np.copy(self.c)

        # Update all point using the diffusion equation as stated in the assignment
        for x in range(0, self.N):
            for y in range(0, self.N):

                # skip cluster values, source values and sink values
                if self.sources[y, x] or self.sinks[y, x] or self.cluster[y, x]:
                    continue

                xmin1 = (x - 1) % self.N
                xplus1 = (x + 1) % self.N
                self.c[y, x] = self.omega / 4 * (self.c[y, xplus1] + self.c[y, xmin1] + self.c[y + 1, x] +
                                                 self.c[y - 1, x]) + (1 - self.omega) * self.c[y, x]

        # Calculate maximum change
        delta = np.max(np.abs(self.c - c_old))

        return delta

    def step_SOR_fast(self):
        """
        Perform one iteration of SOR (self organised relaxation).

        Returns:
            delta: 2d numpy array, the maximum difference between old and new
            concentration arrays
        """
        c_old = np.copy(self.c)

        # Define a convolution kernel for diffusion (4-point stencil) and apply
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]]) / 4
        c_new = convolve(self.c, kernel, mode='wrap')

        # Identify all bitmap positions to be skipped
        mask = self.get_mask()

        # Apply SOR update only to non-bitmap positions
        self.c[~mask] = self.omega * c_new[~mask] + (1 - self.omega) * self.c[~mask]

        delta = np.max(np.abs(self.c - c_old))

        return delta

    def solve_diffusion(self, method=step_SOR):
        """
        Solve the diffusion equation using SOR based on the tolerance level
        of the DLA object.

        Args:
            method: class step function. Supports SOR, either fast version
            or 'normal/slow' version
        """
        delta = method(self)
        while delta > self.tol:
            delta = method(self)

    def update_candidates(self, position):
        """
        Updates the candidates set based on the newly added positions to the
        cluster.

        It removes the positions added to the cluster from the candidates list
        and also adds the neighbours of this position to the candidates lists.

        Returns:
            candidates: updated set of candidate positions: {(int, int)}
        """
        mask = self.get_mask()
        assert position in self.candidates

        self.candidates.remove(position)
        r, c = position
        # loop over all neighbours
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            rn = r + dr
            nc = c + dc
            # add if not a constant position
            if mask[nr, nc] != 1:
                self.candidates.add((nr, nc))

    def calculate_probabilities(self):
        """
        Calculates the probabilities for al the candidates using the formula
        from the assignment.

        Returns:
            probabilities: dictionary of candidate positions to probability:
            {(row: Int, col: Int)} -> probability: Int
        """
        assert len(self.candidates) > 0, "we don't have any candidates"

        probabilities = {}

        # we can get negative concentrations inside of completely surrounded
        # cluster points? So I gues we just clip all concentrations to 0 if < 0?
        for r, c in self.candidates:
            self.c[r, c] = max(0, self.c[r, c])

        norm = sum([self.c[candidate] ** self.eta for candidate in self.candidates])
        for candidate in self.candidates:
            p = max(0, self.c[candidate] ** self.eta / norm)
            assert p >= 0, f"negative probability: norm: {norm}, c: {self.c[candidate]}, candidate: {candidate}"
            probabilities[candidate] = p

        probs = list(probabilities.values())
        assert np.isclose(sum(probs), 1.0, atol=1e-5), "The probabilities do not sum up to 1. " + str(sum(probs))

        return probabilities

    def update_cluster(self, probabilities):
        """
        Update the cluster based on the probabilities determined in
        'calculate_probabilities'.

        It does so by going over all the (position, probabilities) pairs and
        drawing random numbers for each such pair. Then when the random number
        is smaller than the probability we add the point to the cluster.

        Returns:
            positions: list of positions added to the cluster
            [(row: Int, col: Int)]
        """
        probabilities = self.calculate_probabilities()

        # split the positions and the probabilities (python trick)
        poss, probs = zip(*list(probabilities.items()))

        # pick the positions to add:
        pos_to_add_idx = np.random.choice(len(poss), replace=False, p=probs)
        pos_to_add = poss[pos_to_add_idx]

        # make sure position is not outside of the grid
        assert 0 <= pos_to_add[0] < self.N and 0 <= pos_to_add[1] < self.N, f"position: {pos_to_add} is outside of the grid"

        self.cluster[pos_to_add] = 1      # add to cluster
        self.c[pos_to_add] = 0            # set concentration to 0

        return pos_to_add

    def step(self):
        """
        Iterates one step of DLA algorithm.

        A step contains of
        (1) solving the diffusion equation (SOR)
        (2) determining the candidates
        (3) determining the growth probabilities for all candidates
        (4) add a single growth candidate to the cluster
        (5) repeat
        """
        # solves the diffustion equation using SOR
        self.solve_diffusion()

        # 'initialises the candidates' TODO: Change this to update the candidates intead, changes from O(N^2) to O(1)
        self.candidates = self.init_candidates()

        # calculates the probabilities
        probabilities = self.calculate_probabilities()

        # add a single growth candidate to the cluster using the probabilities
        self.update_cluster(probabilities)

        self.NO_steps += 1
        print(f'step: {self.NO_steps}')
        #TODO: ASSERT FOR WHEN FULL, SO IT DOESNT GO OUT OF INDEX (boundary conditions?)

    def plot(self, title="test"):
        """Plot the current state of the system as a 2D color map"""
        plt.figure(figsize=(8, 8))

        im = plt.imshow(self.c,
                        extent=[0, 1,  0, 1],
                        origin='lower',
                        cmap='Spectral',
                        aspect='equal',
                        vmin=0, vmax=1)

        # Overlay cluster
        if self.cluster.any():
            object_color = 'gray'
            cluster_array = np.ma.masked_where(self.cluster == 0, self.cluster)
            plt.imshow(cluster_array,
                       extent=[0, 1, 0, 1],
                       origin='lower',
                       cmap=object_color,
                       alpha=0.5)

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Concentration', fontsize=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.xlabel('x', fontsize=LABELSIZE)
        plt.ylabel('y', fontsize=LABELSIZE)
        plt.yticks(fontsize=TICKSIZE)
        plt.xticks(fontsize=TICKSIZE)
        plt.tight_layout()
        plt.savefig(f'../figures/{title}.pdf')
        plt.show()

    def animate(self, num_frames=200, interval=100, steps_per_frame=1, title=""):
        """Animate the evolution of the system.

        Args:
            num_frames: Total number of animation frames
            interval: Time between frames in milliseconds
            steps_per_frame: Number of diffusion steps calculated per frame
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Main concentration field
        im = ax.imshow(self.c,
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    cmap='Spectral',
                    aspect='equal',
                    vmin=0, vmax=1)

        # Cluster overlay (initially empty but will be updated)
        im_cluster = ax.imshow(np.ma.masked_where(self.cluster == 0, self.cluster),
                            extent=[0, 1, 0, 1],
                            origin='lower',
                            cmap='gray',
                            alpha=0.5)

        # Colorbar
        ax.set_xlabel('x', fontsize=LABELSIZE)
        ax.set_ylabel('y', fontsize=LABELSIZE)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label='Concentration')
        cbar.set_label('Concentration', fontsize=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.yticks(fontsize=TICKSIZE)
        plt.xticks(fontsize=TICKSIZE)

        def update(frame):
            """Update function for the animation."""
            # Do multiple steps per frame
            for _ in range(steps_per_frame):
                self.step()

            # Update the concentration field
            im.set_array(self.c)

            # Update the cluster overlay
            im_cluster.set_array(np.ma.masked_where(self.cluster == 0, self.cluster))

            return [im, im_cluster]

        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames,
                            interval=interval, blit=False)

        # Save animation with timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anim.save(filename=f"../figures/{title}timedep_diffusion_{timestamp}.mkv", writer="ffmpeg")
        #plt.show()

        return anim

"""
questions:
    1) concentratinos can become negative because it could happen that positions
    get surrounded by the cluster, which means that the value for the neighbours
    will be 0, and because of the sor method the concentration of the previous
    time step will be substracted. So then 0 - positive --> negative.
    What do I do when this happens? Right now I'm simply clipping all the
    concentrations of the candidates.

    2) This method is very slow, so how do I run more growth runs?
    3) What results do I look at when varying the eta parameter?
    4) Do I also look at different starting conditions?
    5) for quantifying the influence of omega; do I optimise omega per
    iteration? Or do I look at a few different runs?

    A bit of vagueness in how to measure things
"""