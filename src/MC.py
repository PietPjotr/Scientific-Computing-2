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

class MC:
    def __init__(self, N, ps=1):
        self.N = N
        self.ps = ps        # sticking probability

        # initialize cluster and agents list
        self.cluster = np.zeros((N, N))
        self.agents = []                # list of positions
        self.grid = np.zeros((N, N))

        self.cluster[0, N // 2] = 1    # starting point cluster in the middle

        self.NO_steps = 0

    def spawn_agent(self):
        """
        Spawns an agent at the top row.
        """
        c = np.random.randint(low=0, high=self.N)
        r = self.N - 1
        self.agents.append((r, c))

    def neighbours_the_cluster(self, position):
        """
        Checks if the position neighbours the cluster.

        Returns:
            True if position neighbours the cluster, otherwise False
        """
        nr, nc = position
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            neighr = nr + dr
            neighc = nc + dc

            if 0 <= neighr <= self.N - 1 and 0 <= neighc <= self.N - 1 and \
               self.cluster[neighr, neighc] == 1:
                return True
        return False

    def update_grid(self, old_agents, new_agents):
        """
        Updates the agend grid for plotting purposes.
        """
        for r, c in old_agents:
            self.grid[r, c] = 0
        for nr, nc in new_agents:
            self.grid[nr, nc] = 1

    def get_valid_directions(self, position):
        r, c = position

        # filter the possible random directions by cluster presence
        all_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dirs = []
        for dr, dc in all_dirs:
            nr = r + dr
            nc = (c + dc) % (self.N - 1)

            # if position is valid, the position can't be a cluster position
            if 0 <= nr <= self.N - 1 and self.cluster[nr, nc] != 1:
                dirs.append((dr, dc))

            # if not valid position, the agent is allowed to move there
            # (but will respawn at spawning point)
            elif not 0 <= nr <= self.N - 1:
                dirs.append((dr, dc))
        return dirs

    def update_agents(self):
        """
        Updates the agents accordingly.

        Firstly, releases one agent per 'time step'. Then, moves all the agents
        in a random direction, ensuring periodic boundaries left and right. For
        top and bottom, the agents are removed and spawned back at the top.
        Lastly, also checks if agents should be added to the cluster and removed
        from the agents set.
        """
        self.spawn_agent()
        new_agents = []

        for r, c in self.agents:
            # pick 'random' direction and add to current agent
            dirs = self.get_valid_directions((r, c))
            dr, dc = dirs[np.random.choice(len(dirs))]
            nr = r + dr
            nc = (c + dc) % (self.N - 1)    # wrap around on left and right side

            # check if we don't move outside of the boundary if we do, spawn a
            # new agent
            if not 0 <= nr <= self.N - 1:
                self.spawn_agent()
                continue

            # if neighbouring to cluster, add with probability ps
            if self.neighbours_the_cluster((nr, nc)) and np.random.random() < self.ps:
                assert self.cluster[nr, nc] != 1
                self.cluster[nr, nc] = 1

            # if not neighbours to cluster of gets added, simply update position
            else:
                assert 0 <= nr <= self.N - 1 and 0 <= nc <= self.N - 1, \
                       f"not valid new position {(nr, nc)}"
                new_agents.append((nr, nc))

        # update grid for plotting
        self.update_grid(self.agents, new_agents)

        # update the new agents
        self.agents = new_agents

    def step(self):
        self.update_agents()

        self.NO_steps += 1
        print(f'step: {self.NO_steps}')

    def plot(self, title="test_MC"):
        """Plot the current state of the system as a 2D color map"""
        plt.figure(figsize=(8, 8))

        im = plt.imshow(self.grid,
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

    def animate(self, num_frames=1000, interval=10, steps_per_frame=1, title=""):
        """Animate the evolution of the system.

        Args:
            num_frames: Total number of animation frames
            interval: Time between frames in milliseconds
            steps_per_frame: Number of diffusion steps calculated per frame
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Main concentration field
        im = ax.imshow(self.grid,
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
            im.set_array(self.grid)

            # Update the cluster overlay
            im_cluster.set_array(np.ma.masked_where(self.cluster == 0, self.cluster))

            return [im, im_cluster]

        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames,
                            interval=interval, blit=False)

        # Save animation with timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anim.save(filename=f"../figures/MC_{timestamp}.mkv", writer="ffmpeg")
        return anim
