from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pso import PSOMultiprocessing
from utils import get_divs, get_tr_te_subsets, make_comparations, save, get_path
from bms import BMS
from srm import SRM


# PATH = 'datasets/iris.data'
# CL_COL = 4

PATH = 'datasets/wine.data'
CL_COL = 0


def main(save_res=False):
    if save_res:
        start_time = datetime.now().strftime('%H:%M:%S')

    dataset = pd.read_csv(PATH, header=None).sort_values(by=CL_COL, ignore_index=True).values
    fe_cols = (1, dataset.shape[1]) if CL_COL == 0 else (0, dataset.shape[1]-1)
    divs = get_divs(dataset, CL_COL)
    class_names = tuple(dataset[divs[i]][CL_COL] for i in range(len(divs)-1))

    sn_model = BMS()
    # sn_model = SRM()
    n_model_parms = dict(filter(lambda i: not isinstance(i[1], list), sn_model.__dict__.items()))

    assert 'get_firing_trace' in dir(sn_model), 'La clase para el modelo neuronal tiene que \
implementar el metodo \'get_firing_trace\''
    assert 'run' in dir(sn_model), 'La clase para el modelo neuronal tiene que implementar \
el metodo \'run\''


    # ***** Setting up training and testing subsets *****
    tr_features, te_features = get_tr_te_subsets(dataset[:, fe_cols[0]:fe_cols[1]], divs)

    del divs, fe_cols


    # ***** Configure model *****
    weights, _ = PSOMultiprocessing(tr_features, sn_model, save_plot=save_res).run()
    # weights = [-0.42493396, -0.21774958,  1.18442043,  0.52923564 ] # iris
    # weights = [1.03399134, 0.73495792, 0.86025217, 1.71217792, 0.69568314, 0.3161272,
    # 1.37551808, 0.91079394, 0.69921659, 1.00514106, 0.40010956, 1.26970743, 0.8569052] # wine
    print('weights =', weights, end=' - ')

    afrs = np.empty(len(tr_features), dtype=np.float64)

    if save_res:
        cl_track = 1
        colors = ['red', 'blue', 'green', 'black', 'orange', 'brown', 'yellow']
        plt.clf()
        # Compute average firing rates and plot firing trace
        for cl_idx, cl in enumerate(tr_features):
            firing_rates = np.empty(cl.shape[0], dtype=np.float64)

            # For each element
            for i, e in enumerate(cl):
                firing_trace, firing_rates[i] = sn_model.get_firing_trace(np.dot(e, weights))

                # Graph firing trace
                plt.scatter(firing_trace, np.full(firing_trace.shape[0], cl_track+i,
                            dtype=np.int16), c=colors[cl_idx%len(colors)], s=2)

            cl_track += cl.shape[0]
            afrs[cl_idx] = np.mean(firing_rates)

        plt.savefig(get_path('firing_trace.png'))

        del cl_track, firing_trace, colors

    else:
        # Compute average firing rates
        for cl_idx, cl in enumerate(tr_features):
            firing_rates = np.empty(cl.shape[0], dtype=np.float64)

            # For each element
            for i, e in enumerate(cl):
                firing_rates[i] = sn_model.run(np.dot(e, weights))

            afrs[cl_idx] = np.mean(firing_rates)

    print('afrs =', afrs)

    del firing_rates


    # ***** Comparations *****
    # With training subset
    predics = [[np.argmin([abs(fr-afr) for afr in afrs]) for fr in (sn_model.run(
                np.dot(e, weights)) for e in cl)] for cl in tr_features]
    tr_res_str = make_comparations('Training', predics, class_names)
    print(tr_res_str)

    # With testing subset
    predics = [[np.argmin([abs(fr-afr) for afr in afrs]) for fr in (sn_model.run(
                np.dot(e, weights)) for e in cl)] for cl in te_features]
    te_res_str = make_comparations('Testing', predics, class_names)
    print(te_res_str)


    # ***** Save results *****
    if save_res:
        save(start_time=start_time, dataset=re.search(r'/(\S+\.\S+$)', PATH).group(1),
            sn_model=sn_model.__class__.__name__, neural_model_parameters=n_model_parms,
            weights=weights, afr=afrs, training_results=tr_res_str, testing_results=te_res_str,
            finish_time=datetime.now().strftime('%H:%M:%S'))


if __name__ == '__main__':
    main(save_res=False)
