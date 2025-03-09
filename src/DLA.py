"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Contains the implementation of the Diffusion Limited Aggregation
(DLA) model and related analysis functions. This file makes use of SOR for
the numerical diffusion solver. It also contains some plot and animate and
store functions to store the results of the experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


# global vars indicated by all caps
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)

LABELSIZE = 20
TICKSIZE = 16


class DLA:
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
        self.cluster[0, N // 2] = 1         # starting point cluster in the middle

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
            nr = r + dr
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

        # Break if the cluster reaches the top of the grid
        if np.any(self.cluster[self.N -2, :] == 1):
            print(f"Cluster reached the top of the grid at step {self.NO_steps}")
            raise StopIteration

    def save_as_csv(self):
        """Save the current concentration field to a CSV file."""
        filename = f"../results/DLA_concentration_eta{self.eta}.csv"
        np.savetxt(filename, self.c, delimiter=",")
        print(f"Concentration field saved to {filename}")
        cluster_filename = f"../results/DLA_cluster_eta{self.eta}.csv"
        np.savetxt(cluster_filename, self.cluster, delimiter=",")
        print(f"Cluster field saved to {filename}")

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

        anim.save(filename=f"../figures/{title}DLA.mkv", writer="ffmpeg")
        plt.show()

        return anim

    def analyze_dla(DLA_class, eta_values=[0.5, 1.0, 1.5, 2.0], N=100, max_steps=200, omega=1.9):
        """Compact analysis of DLA growth and convergence for different eta values"""
        results = {}

        for eta in eta_values:
            print(f"\nAnalyzing eta = {eta}")

            # Create DLA instance
            dla = DLA_class(N=N, eta=eta, tol=1e-4, omega=omega)

            # Track metrics
            heights = []          # Max height at each step
            sizes = []            # Number of cluster sites
            iterations = []       # SOR iterations per step

            start_time = time.time()

            # Run simulation
            for step in range(max_steps):
                # Record cluster metrics
                cluster_points = np.argwhere(dla.cluster == 1)
                sizes.append(len(cluster_points))
                heights.append(np.max(cluster_points[:, 0]) if len(cluster_points) > 0 else 0)

                # Solve diffusion with iteration counting
                iters = 0
                delta = dla.step_SOR_fast()
                iters += 1
                while delta > dla.tol:
                    delta = dla.step_SOR_fast()
                    iters += 1
                iterations.append(iters)

                # Update candidates and grow cluster
                dla.candidates = dla.init_candidates()
                probabilities = dla.calculate_probabilities()
                dla.update_cluster(probabilities)
                dla.NO_steps += 1

                # Print progress occasionally
                if step % 20 == 0:
                    print(f"  Step {step}, Height: {heights[-1]}, Iterations: {iters}")

                # Stop if cluster reaches top
                if np.any(dla.cluster[N-2, :] == 1):
                    print(f"  Cluster reached top at step {step}")
                    break

            total_time = time.time() - start_time

            # Store results
            results[eta] = {
                'cluster': dla.cluster.copy(),
                'heights': np.array(heights),
                'sizes': np.array(sizes),
                'iterations': np.array(iterations),
                'steps': len(heights),
                'time': total_time
            }

            # Print summary
            print(f"  Steps completed: {len(heights)}")
            print(f"  Final height: {heights[-1]}")
            print(f"  Average iterations: {np.mean(iterations):.2f}")
            print(f"  Total time: {total_time:.2f} seconds")

        return results

    def plot_results(results):
        """Create concise plots of key metrics"""
        eta_values = sorted(results.keys())
        colors = sns.color_palette("viridis", len(eta_values))

        # Figure 1: Growth metrics
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Height growth
        for i, eta in enumerate(eta_values):
            steps = np.arange(len(results[eta]['heights']))
            ax[0].plot(steps, results[eta]['heights'], label=f'η = {eta}', color=colors[i])

        ax[0].set_xlabel('Growth Steps')
        ax[0].set_ylabel('Maximum Height')
        ax[0].set_title('Cluster Height Growth')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        # Height vs Size (density)
        for i, eta in enumerate(eta_values):
            ax[1].plot(results[eta]['sizes'], results[eta]['heights'],
                    label=f'η = {eta}', color=colors[i])

        ax[1].set_xlabel('Cluster Size (sites)')
        ax[1].set_ylabel('Maximum Height')
        ax[1].set_title('Height vs Size (Density)')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig('dla_growth_metrics.pdf')
        plt.show()

        # Figure 2: Convergence metrics
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Iterations per step
        for i, eta in enumerate(eta_values):
            steps = np.arange(len(results[eta]['iterations']))
            ax[0].plot(steps, results[eta]['iterations'], label=f'η = {eta}', color=colors[i])

        ax[0].set_xlabel('Growth Steps')
        ax[0].set_ylabel('SOR Iterations')
        ax[0].set_title('Convergence Iterations')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        # Average iterations by eta
        avg_iters = [np.mean(results[eta]['iterations']) for eta in eta_values]
        ax[1].bar(range(len(eta_values)), avg_iters, color=colors)
        ax[1].set_xlabel('η Value')
        ax[1].set_ylabel('Average Iterations')
        ax[1].set_title('Average SOR Iterations')
        ax[1].set_xticks(range(len(eta_values)))
        ax[1].set_xticklabels([str(eta) for eta in eta_values])
        ax[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('dla_convergence_metrics.pdf')
        plt.show()

    def run_analysis(DLA_class):
        """Run DLA analysis and plot results"""
        # Define parameters
        eta_values = [1, 3, 5, 8, 10]
        N = 100
        max_steps = 200
        omega = 1.9

        print(f"Running DLA analysis for eta values: {eta_values}")
        results = analyze_dla(DLA_class, eta_values, N, max_steps, omega)
        plot_results(results)

        # Summary table
        print("\n===== Results Summary =====")
        print(f"{'Eta':^5} | {'Steps':^6} | {'Height':^6} | {'Size':^6} | {'Avg Iter':^8} | {'Time (s)':^8}")
        print("-" * 50)

        for eta in sorted(results.keys()):
            r = results[eta]
            print(f"{eta:^5} | {r['steps']:^6} | {r['heights'][-1]:^6.0f} | {r['sizes'][-1]:^6} | "
                  f"{np.mean(r['iterations']):^8.2f} | {r['time']:^8.2f}")

        return results
