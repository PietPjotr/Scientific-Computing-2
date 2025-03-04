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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve      # used for fast double for loop
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime
import json
import os

# global vars indicated by all caps
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)

LABELSIZE = 20
TICKSIZE = 16

class MC:
    def __init__(self, N, ps=1, max_iter=1000, timestamp=None):
        self.N = N
        self.ps = ps        # sticking probability
        self.max_iter = max_iter

        # initialize cluster and agents list
        self.cluster = np.zeros((N, N))
        self.grid = np.zeros((N, N))    # grid with agents
        self.agents = []                # list of positions
        self.spawn_agent()

        self.cluster[0, N // 2] = 1     # starting point cluster in the middle

        self.NO_steps = 0
        if not timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # used for parallel storing
        else:
            self.timestamp = timestamp

    def spawn_agent(self):
        """
        Spawns an agent at the top row.
        """
        c = np.random.randint(low=0, high=self.N)
        r = self.N - 1
        self.agents.append((r, c))
        self.grid[r, c] = 1

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

    def get_valid_directions(self, position):
        r, c = position

        # filter the possible random directions by cluster presence
        all_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dirs = []
        for dr, dc in all_dirs:
            nr = r + dr
            nc = (c + dc) % (self.N - 1)

            # if position is valid, the position can't be a cluster position
            if 0 <= nr <= self.N - 1 and self.cluster[nr, nc] != 1 and self.grid[nr, nc] != 1:
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
        new_agents = []
        # we cap the amount of agents to 1% density
        if len(self.agents) < self.N:
            self.spawn_agent()

        for r, c in self.agents:
            # pick 'random' direction and add to current agent
            dirs = self.get_valid_directions((r, c))
            if len(dirs) > 0:
                dr, dc = dirs[np.random.choice(len(dirs))]
            else:
                dr, dc = 0, 0
            nr = r + dr
            nc = (c + dc) % (self.N - 1)    # wrap around on left and right side

            # check if we don't move outside of the boundary if we do, spawn a
            # new agent
            if not 0 <= nr <= self.N - 1:
                self.spawn_agent()
                self.grid[r, c] = 0
                continue

            # if neighbouring to cluster, add with probability ps
            if self.neighbours_the_cluster((nr, nc)) and np.random.random() < self.ps:
                assert self.cluster[nr, nc] != 1
                self.cluster[nr, nc] = 1
                self.spawn_agent()
                self.grid[r, c] = 0

            # if not neighbours to cluster of gets added, simply update position
            else:
                assert 0 <= nr <= self.N - 1 and 0 <= nc <= self.N - 1, \
                       f"not valid new position {(nr, nc)}"
                new_agents.append((nr, nc))

                # update the grid for the animation
                self.grid[r, c] = 0
                self.grid[nr, nc] = 1

        # update the new agents
        self.agents = new_agents

    def step(self):
        self.update_agents()

        self.NO_steps += 1
        if self.NO_steps % 10000 == 0:
            print(f'step: {self.NO_steps}')

        # Break if the cluster reaches the top of the grid
        if np.any(self.cluster[self.N - 2, :] == 1) or self.NO_steps >= self.max_iter:
            print(f"Cluster reached the top of the grid at step {self.NO_steps} or \
                    max iterations reached.")
            self.save_to_csv()
            raise StopIteration

    def run(self):
        for i in range(self.max_iter):
            self.step()

    def save_to_csv(self, data_folder="../data/MC"):
        """
        Saves the agent and cluster bitmaps to separate CSV files
        and stores metadata including the sticking probability (ps).
        """
        # Ensure the data directory exists
        data_folder += '/' + str(self.timestamp)
        os.makedirs(data_folder, exist_ok=True)

        # Generate filenames with timestamp and sticking probability
        agent_csv_filename = os.path.join(data_folder, f"MC_ps{self.ps}_agent.csv")
        cluster_csv_filename = os.path.join(data_folder, f"MC_ps{self.ps}_cluster.csv")
        metadata_filename = os.path.join(data_folder, f"MC_ps{self.ps}_metadata.json")

        # Save the bitmaps to CSV
        np.savetxt(agent_csv_filename, self.grid, delimiter=",", fmt="%d")
        np.savetxt(cluster_csv_filename, self.cluster, delimiter=",", fmt="%d")

        # Save metadata
        metadata = {
            "sticking_probability": self.ps,
            "agent_csv": agent_csv_filename,
            "cluster_csv": cluster_csv_filename
        }

        with open(metadata_filename, "w") as metafile:
            json.dump(metadata, metafile)

        print(f"Agent bitmap saved to {agent_csv_filename}")
        print(f"Cluster bitmap saved to {cluster_csv_filename}")
        print(f"Metadata saved to {metadata_filename}")
