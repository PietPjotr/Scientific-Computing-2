import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = sns.color_palette("Set2", 8)

LABELSIZE = 20
TICKSIZE = 16

def load_csv(filename):
    return np.loadtxt(f"../results/{filename}", delimiter=",")

def plot_results(filenames, title="GrayScott_comparison"):
    n_cases = len(filenames)
    fig, axs = plt.subplots(n_cases, 1, figsize=(5, 5 * n_cases))
    if n_cases == 1:
        axs = [axs]
    for i, filename in enumerate(filenames):
        data = load_csv(filename)
        im = axs[i].imshow(data, cmap='Spectral', origin='lower', extent=[0, 1, 0, 1])
        axs[i].set_xlabel('x', fontsize=LABELSIZE)
        axs[i].set_ylabel('y', fontsize=LABELSIZE)
        axs[i].tick_params(labelsize=TICKSIZE)
        axs[i].set_title(filename, fontsize=LABELSIZE)
        cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        cbar.set_label('Concentration', fontsize=LABELSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
    plt.tight_layout()
    plt.savefig(f'../figures/GrayScott/{title}.pdf')
    plt.show()
