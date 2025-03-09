"""
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description: Contains the GrayScott model implementation.

Contains the reaction rules for the GrayScott model in a numerical manner.
The class can be initiated with different values for all the relevant
parameters. Also contains an animate function that steps through time as the
object is run.
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Global settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)

LABELSIZE = 20
TICKSIZE = 16


class GrayScott:
    def __init__(self, N=100, Du=0.16, Dv=0.08, f=0.035, k=0.06, dt=1, dx=1, noise=True):
        self.N = N
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.dt = dt
        self.dx = dx

        # Initial conditions
        self.u = np.ones((N, N)) * 0.5  # u0 = 0.5 = initial condition
        self.v = np.zeros((N, N))
        # square in the middle
        middle = N//2
        r = 5
        self.v[middle-r:middle+r, middle-r:middle+r] = 0.25

        # Noise
        if noise:
            self.v += 0.01*np.random.rand(N, N)
            self.u += 0.01*np.random.rand(N, N)

    def Boundary_conditions(self, vector):
        '''
        Periodic boundary conditions
        '''
        vector[0, :] = vector[-2, :]
        vector[-1, :] = vector[1, :]
        vector[:, 0] = vector[:, -2]
        vector[:, -1] = vector[:, 1]

    def Laplacian(self, u):
        '''
        Determine nabla squared, e.g. Laplacian
        '''
        nabla_squared = (u[:-2, 1:-1] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]
                         + u[1:-1, 2:] + u[2:, 1:-1]) / self.dx**2
        return nabla_squared

    def Reaction(self):
        '''
        Reaction step of the Gray-Scott model
        '''
        u_ins, v_ins = self.u[1:-1, 1:-1], self.v[1:-1, 1:-1]
        Lapl_u = self.Laplacian(self.u)
        Lapl_v = self.Laplacian(self.v)

        uv_squared = u_ins*v_ins*v_ins
        dudt = self.Du*Lapl_u - uv_squared + self.f*(1 - u_ins)
        dvdt = self.Dv*Lapl_v + uv_squared - (self.f + self.k) * v_ins

        self.u[1:-1, 1:-1] += dudt * self.dt
        self.v[1:-1, 1:-1] += dvdt * self.dt

        self.Boundary_conditions(self.u)
        self.Boundary_conditions(self.v)

    def run_until_criterium(self, threshold=0.01, max_steps=10000):
        '''
        Runs simulation until the threshhold concentration or max_steps is reached.
        '''
        step = 0
        while step < max_steps:
            self.Reaction()
            if np.all(self.v < threshold):
                print(f"All concentrations in v fell below {threshold} at step {step}.")
                break
            step += 1
        return step

    def plot(self, title="Gray-ScottSimulation"):
        """Plot the current state of the system."""
        plt.figure(figsize=(8, 8))
        im = plt.imshow(self.v, cmap='Spectral', origin='lower', extent=[0, 1, 0, 1])
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Concentration', fontsize=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.xlabel('x', fontsize=LABELSIZE)
        plt.ylabel('y', fontsize=LABELSIZE)
        plt.xticks(fontsize=TICKSIZE)
        plt.yticks(fontsize=TICKSIZE)
        plt.tight_layout()
        plt.savefig(f'../figures/GrayScott/{title}.pdf')

    def animate(self, num_frames=200, interval=100, steps_per_frame=1, title="GrayScott"):
        """Animate the evolution of the system."""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.v, cmap='Spectral', origin='lower', extent=[0, 1, 0, 1])
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Concentration', fontsize=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.xticks(fontsize=TICKSIZE)
        plt.yticks(fontsize=TICKSIZE)

        def update(frame):
            for _ in range(steps_per_frame):
                self.Reaction()
            im.set_array(self.v)
            return [im]

        anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
        anim.save(f'../figures/GrayScott/{title}.mkv', writer="ffmpeg")
        return anim

    def save_to_csv(self, title="GrayScott"):
        np.savetxt(f'../results/{title}.csv', self.v, delimiter=',')
        print(f"Data saved to ../results/{title}.csv")
