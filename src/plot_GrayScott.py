import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

LABELSIZE = 26
TICKSIZE = 24

def load_csv(filename):
    '''
    Load csv file into numpy array
    '''
    return np.loadtxt(f"../results/{filename}", delimiter=",")

def plot_results(filenames, variable_name, variables, iterations, title="GrayScott_comparison"):
    '''	
    Makes subplots of the Gray-Scott model results for different values of a parameter.
    ''' 
    n_cases = len(filenames)
    n_cols = 2
    n_rows = int(np.ceil(n_cases / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for i, filename in enumerate(filenames):
        data = load_csv(filename)
        im = axs[i].imshow(data, cmap='Spectral', origin='lower', extent=[0, 1, 0, 1])
        axs[i].set_title(rf"${variable_name}$ = {variables[i]}, {iterations[i]} iterations", fontsize=LABELSIZE, pad=15)
        axs[i].tick_params(labelsize=TICKSIZE)
        cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        if i % n_cols == 0:
            axs[i].set_ylabel('y', fontsize=LABELSIZE)
        else:
            cbar.set_label('Concentration', fontsize=LABELSIZE)
        if i == n_cases - 1 or i == n_cases - 2:
            axs[i].set_xlabel('x', fontsize=LABELSIZE)
            axs[i].tick_params(labelbottom=True)
    
    # Remove any extra axes if n_cases is less than the total subplots.
    for j in range(n_cases, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig(f'../figures/GrayScott/{title}.pdf')