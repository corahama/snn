import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import get_divs, norm_features


DATASET = pd.read_csv('datasets/iris.data', header=None).values
CL_COL = 4

# DATASET = pd.read_csv('datasets/wine.data', header=None).values
# CL_COL = 0

FE_COLS = (1, DATASET.shape[1]) if CL_COL == 0 else (0, DATASET.shape[1]-1)
DIVS = get_divs(DATASET, CL_COL)


"""Function to compute feature means by class, and the summation of them"""
def compute_fe_means():
    summation_list = []
    for i in range(len(DIVS)-1):
        summation = 0
        print(f'Results for class \'{DATASET[DIVS[i], CL_COL]}\': ', end='\t')
        for j in range(*FE_COLS):
            mean = DATASET[DIVS[i]:DIVS[i+1], j].mean()
            std = DATASET[DIVS[i]:DIVS[i+1], j].std()
            summation += mean
            print(f'({mean:.2f},{std:.2f})-', end='')
            # print(f'{mean:.2f}-', end='')
        summation_list.append(summation)
        print(f'Total: {summation:.2f}')

    return summation_list

"""Function to compute weights of feature means in global summation"""
def compute_weights(summation_list):
    for i, s in zip(range(len(DIVS)-1), summation_list):
        print(','.join(f'{(DATASET[DIVS[i]:DIVS[i+1], j].mean()/s):.3f}' for j
            in range(*FE_COLS)))

"""Print firing trains obtain by dataset patterns encoding"""
def print_firing_train():
    # number of neurons per variable, and beta constant
    m, b = 4, 2

    # gauss activation function
    gauss = lambda x, mean, sd: str(int(np.around((1-np.exp(-(x-mean)**2/(2*sd**2)))*10))).zfill(2)

    # set rf parameters
    rf_data = []
    for j in range(*FE_COLS):
        fe_data = DATASET[:, j].astype(np.float64)
        min_val, max_val = fe_data.min(), fe_data.max()

        for i in range(1, m+1):
            rf_data.append((min_val+((2*i-3)/2)*((max_val-min_val)/(m-2)),
                (1/b)*((max_val-min_val)/(m-2))))

    # print spike train per pattern
    for p_idx, p in enumerate(DATASET):
        print(str(p_idx+1).zfill(3), end=' ')
        for fe_idx, fe in enumerate(p[FE_COLS[0]:FE_COLS[1]]):
            print(f'({"-".join(gauss(fe, *rf_data[fe_idx*m+i]) for i in range(m))})', end=' ')
        print(f'({p[CL_COL]})')

"""Graph gaussian receptive fields generated with dummy parameters"""
def graph_dummy_rf():
    min_v, max_v = -1.5, 1.5
    m = 5
    b = 2

    # Gauss activation function
    gauss = lambda x: np.exp(-(x-mean)**2/(2*sd**2))

    for i in range(1, m+1):
        mean = min_v + ((2*i-3)/2)*((max_v-min_v)/(m-2))
        sd = (1/b)*((max_v-min_v)/(m-2))

        x = np.arange(np.floor(min_v), np.ceil(max_v), .01)
        y = tuple(map(gauss, x))
        plt.plot(x, y)
    plt.grid()
    plt.show()

"""Graph gaussian receptive fields generated with ranges of dataset features"""
def graph_rf():
    # number of neurons per variable, and beta constant
    m, b = 4, 2

    # Gauss activation function
    gauss = lambda x, mean, sd: np.exp(-(x-mean)**2/(2*sd**2))

    figure, axis = plt.subplots(2, int(np.ceil((FE_COLS[1]-FE_COLS[0])/2)))
    for j in range(*FE_COLS):
        fe_data = DATASET[:, j].astype(np.float64)
        min_val, max_val = fe_data.min(), fe_data.max()

        for i in range(1, m+1):
            mean = min_val + ((2*i-3)/2)*((max_val-min_val)/(m-2))
            sd = (1/b)*((max_val-min_val)/(m-2))

            x = np.arange(np.floor(min_val), np.ceil(max_val), .01)
            y = tuple(map(lambda xi: gauss(xi, mean, sd), x))

            axis[int(j/2), j%2].plot(x, y, label=f'ID {i}')
        axis[int(j/2), j%2].set_title(f'Feature {j+1}')
        axis[int(j/2), j%2].grid()
        axis[int(j/2), j%2].legend()
    plt.show()


def main():
    # summation_list = compute_fe_means()
    # compute_weights(summation_list)

    # norm_features(DATASET, FE_COLS)
    # summation_list = compute_fe_means()
    # compute_weights(summation_list)

    # norm_features(DATASET, FE_COLS)
    # for i in range(len(DIVS)-1):
    #     print(f'class: {DATASET[DIVS[i]][CL_COL]}')
    #     for e in (DATASET[e_idx] for e_idx in range(DIVS[i], DIVS[i+1])):
    #         print(np.dot(e[FE_COLS[0]:FE_COLS[1]], [1]*(FE_COLS[1]-FE_COLS[0])))


    print_firing_train()
    # graph_dummy_rf()
    graph_rf()


    return


if __name__ == '__main__':
    main()
