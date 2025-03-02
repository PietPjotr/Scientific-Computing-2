import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from datetime import datetime

# Global settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)

LABELSIZE = 20
TICKSIZE = 16

class GrayScott:
    def __init__(self, N = 100, Du= 0.16, Dv = 0.08, f = 0.035, k = 0.06, dt = 1, dx = 1):
        self.N = N
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.dt = dt
        self.dx = dx

        # Initial conditions
        self.u = np.ones((N, N)) * 0.5 # u0 = 0.5 = initial condition
        self.v = np.zeros((N, N))
        # square in the middle
        middle = N//2
        r = 5
        self.v[middle-r:middle+r, middle-r:middle+r] = 0.25

        # Noise
        self.v += 0.01*np.random.rand(N, N)



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
        nabla_squared = (u[ :-2, 1:-1] + u[1:-1, :-2] - 4*u[1:-1, 1:-1] 
                        + u[1:-1, 2:] +   u[2:  , 1:-1] )
        return nabla_squared
    
    def Reaction(self):
        '''
        Reaction step of the Gray-Scott model
        '''
        u, v = self.u[1:-1, 1:-1], self.u[1:-1, 1:-1]
        Lapl_u = self.Laplacian(self.u)
        Lapl_v = self.Laplacian(self.v)

        uv_squared = self.u*self.v*self.v
        dudt = self.Du*Lapl_u - uv_squared + self.f*(1 - self.u)
        dvdt = self.Dv*Lapl_v + uv_squared - (self.f + self.k)*self.v

        self.u += dudt
        self.v += dvdt

        self.Boundary_conditions(self.u)
        self.Boundary_conditions(self.v)

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
        plt.savefig(f'../figures/{title}.pdf')
        plt.show()

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
                self.update()
            im.set_array(self.v)
            return [im]
        
        anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anim.save(f'../figures/{title}_{timestamp}.mkv', writer="ffmpeg")
        return anim
        
