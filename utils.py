from multiprocessing import Pool, cpu_count
from math import ceil
from os.path import exists
from os import mkdir
from datetime import date
import re

import numpy as np


"""Function excecuted by each process in pool to calculate class divisions"""
def calc_divs(ds, cl_col, beginning, ck_size):
    divs = []
    cl_name = ds[beginning, cl_col]
    for i, e in enumerate(ds[beginning:beginning+ck_size], start=beginning):
        if e[cl_col] != cl_name:
            divs.append(i)
            cl_name = e[cl_col]

    return divs

"""Calculate dataset divisions by class"""
def get_divs(ds, cl_col):
    ps = cpu_count()
    ds_len = len(ds)
    ck_size = ceil(ds_len/ps)

    with Pool(ps) as pool:
        results = [pool.apply_async(calc_divs, args=(ds, cl_col, i*ck_size, ck_size
                    if (i+1)*ck_size <= ds_len else ds_len%ck_size)) for i in range(ps)]
        pool.close()
        pool.join()

    divs = [0]
    for r in results:
        divs += r.get()
    divs.append(ds_len)

    return divs


"""Normalize features in dataset"""
def norm_features(dataset, fe_cols):
    for j in range(*fe_cols):
        min_val, max_val = min(dataset[:, j]), max(dataset[:, j])
        diff = max_val - min_val
        for i in range(dataset.shape[0]):
            dataset[i][j] = (dataset[i][j]-min_val)*10/diff


"""Generate path for result files"""
def get_path(filename):
    if not exists('results/'):
        mkdir('results/')

    num = 1
    today = date.today()
    file_path = lambda: f'results/{today.strftime("%m-%d-%Y")}\
{f"({num})" if num > 1 else ""} {filename}'

    while exists(file_path()):
        num += 1

    return file_path()

"""Save results into a file"""
def save(**kwargs):

    with open(get_path('data.txt'), 'w', encoding='utf-8') as f:
        for key, value in kwargs.items():
            f.write(f'{key}: {value}\n\n')

    return


"""Read weights and delays from a file"""
def read_wd(filepath):
    f = open(filepath, mode='r', encoding='utf-8')

    line = None
    while line != '':
        line = f.readline()
        match = re.search('^(weights_and_delays:) (.*)', line)
        if match is not None:
            break
    f.close()

    if match is None:
        raise Exception('file doesn\'t have a weights_and_delays value')

    return eval(match.group(2))


"""Divide dataset features into training and testing subsets"""
def get_tr_te_subsets(features, divs):
    # Generate random indices to extract testing elements
    test_idcs = [np.random.choice(np.arange(divs[i], divs[i+1]), int((divs[i+1]-divs[i])*.1),
        False) for i in range(len(divs)-1)]

    # Generate testing subset
    te_features = []
    for idcs in test_idcs:
        te_features.append(features[idcs])

    # Generate training subset
    summation = 0
    for i in range(len(divs)-1):
        summation += test_idcs[i].shape[0]
        divs[i+1] -= summation

    tr_features = np.split(np.delete(features, np.concatenate(test_idcs), axis=0), divs[1: -1])

    return tr_features, te_features

"""Compare actual pattern labels to predicted labels and return a string with the results"""
def make_comparations(subset_name, predictions, class_names):
    res_str = f'\n****{subset_name} Subset Results****'

    total_acc = 0
    for cl_idx, cl_predics in enumerate(predictions):

        accuracy = 0
        for predic in cl_predics:
            if predic == cl_idx:
                accuracy += 1
        total_acc += accuracy

        res_str += f'\n· Class \'{class_names[cl_idx]}\': {accuracy}/{len(cl_predics)}'

    res_str += f'\nTotal accuracy: {100*total_acc/sum([len(cl) for cl in predictions]):.2f}%'

    return res_str


if __name__ == '__main__':
    import pandas as pd

    name = 'fernando'
    age = 25
    hobbies = ['programming', 'read', 'gym']

    di = {'degree': 'IT', 'languages': ['python', 'js', 'java']}

    save(name=name, age=age, hobbies=hobbies, **di)
