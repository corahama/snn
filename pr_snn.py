from datetime import datetime
import re

import numpy as np
import pandas as pd

from snn import SNN
from genetic_algorithm import NNGA
from utils import get_divs, get_tr_te_subsets, make_comparations, save, read_wd


PATH = 'datasets/iris.data'
CL_COL = 4


def main(save_res=False):
    if save_res:
        start_time = datetime.now().strftime('%H:%M:%S')

    dataset = pd.read_csv(PATH, header=None).values
    fe_cols = (1, dataset.shape[1]) if CL_COL == 0 else (0, dataset.shape[1]-1)
    divs = get_divs(dataset, CL_COL)
    class_names = tuple(dataset[divs[i]][CL_COL] for i in range(len(divs)-1))

    # ***** Setting up training and testing subsets *****
    tr_features, te_features = get_tr_te_subsets(dataset[:, fe_cols[0]:fe_cols[1]], divs)

    del divs

    # ****** Definition of neural network ******
    m = 4
    h_lsize = 10
    fe_ranges = tuple((col.min(), col.max()) for col in (dataset[:, j] for j in range(*fe_cols)))
    snn = SNN(fe_ranges, m, h_lsize)

    # Run training step to obtain optimized weights and delays
    wd = NNGA(snn, tr_features).run(100, graph=save_res)

    # Read weights and delays from a file
    # wd = read_wd('results/06-21-22 data.txt')

    snn.set_neural_synapses(wd)

    avg_ftimes = []
    for cl in tr_features:
        ftimes = []
        for e in cl:
            ftimes.append(snn.simulate(e))
        avg_ftimes.append(np.mean(ftimes))

    # ***** Comparations *****
    # With training subset
    predics = [[np.argmin([abs(ft-aft) for aft in avg_ftimes]) for ft in (snn.simulate(e)
                for e in cl)] for cl in tr_features]
    tr_res_str = make_comparations('Training', predics, class_names)
    print(tr_res_str)

    # With testing subset
    predics = [[np.argmin([abs(ft-aft) for aft in avg_ftimes]) for ft in (snn.simulate(e)
                for e in cl)] for cl in te_features]
    te_res_str = make_comparations('Testing', predics, class_names)
    print(te_res_str)

    # ***** Save results *****
    if save_res:
        save(start_time=start_time, dataset=re.search(r'/(\S+\.\S+$)', PATH).group(1),
            weights_and_delays=wd, average_firing_times=avg_ftimes, training_results=tr_res_str,
            testing_results=te_res_str, finish_time=datetime.now().strftime('%H:%M:%S'))


if __name__ == '__main__':
    main(save_res=True)
