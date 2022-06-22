from itertools import count
from sys import maxsize
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

from utils import get_path


def binary_search(arr, num):
    low, high = 0, len(arr)-2

    while low <= high:
        mid = (high+low)//2

        if arr[mid] > num:
            high = mid-1
        elif arr[mid] <= num:
            if num < arr[mid+1]:
                return mid
            low = mid+1

    return None


"""Genetic Algorithm for minimization optimization"""
class GA():
    def __init__(self, pop_size=100, str_size=20):
        # Initialization
        self.population = np.random.randint(2, size=(pop_size, str_size), dtype=bool)
        self.pop_fitnesses = None
        self.best = maxsize
        self.idx_best_solution = None

    def run(self, iter_num=50, graph=False):
        self.pop_fitnesses = self.calculate_fitnesses(self.population)

        history = np.empty(shape=iter_num, dtype=np.float64)
        for it in range(iter_num):
            # Selection
            pop_fitnesses = np.array([1/fit if fit != 0 else maxsize for fit in self.pop_fitnesses])
            sum_pop_fits = sum(pop_fitnesses)

            prob_distribution = []
            summation = 0
            for fit in pop_fitnesses:
                prob_distribution.append(summation)
                summation += fit/sum_pop_fits
            prob_distribution.append(1)
            prob_distribution = np.array(prob_distribution, dtype=np.float64)

            selection = []
            for _ in range(self.population.shape[0]):
                rand_num = np.random.rand()
                selection.append(self.population[binary_search(prob_distribution, rand_num)])
            selection = np.array(selection, dtype=bool)

            # Crossover
            for i in range(0, selection.shape[0]-len(selection)%2, 2):
                c_point = np.random.randint(1, selection.shape[1])
                temp_arr = selection[i][c_point::].copy()
                selection[i][c_point::] = selection[i+1][c_point::]
                selection[i+1][c_point::] = temp_arr

            # Mutation (5% of chromosome)
            for chromosome in selection:
                idcs = np.random.randint(0, selection.shape[1], size=int(selection.shape[1]*.05))
                for i in idcs:
                    chromosome[i] = not chromosome[i]

            # Evolution and replacement
            sel_fitnesses =  self.calculate_fitnesses(selection)
            for i, sel_fit, pop_fit in zip(count(), sel_fitnesses, self.pop_fitnesses):
                if sel_fit < pop_fit:
                    self.population[i] = selection[i]
                    self.pop_fitnesses[i] = sel_fitnesses[i]

            # Save best value in history
            for i, pop_fit in enumerate(self.pop_fitnesses):
                if pop_fit < self.best:
                    self.best = pop_fit
                    self.idx_best_solution = i
            history[it] = self.best
            print(f'best in iteration {it+1}: {self.best}')

        # Graph history
        if graph:
            plt.plot(np.arange(1, iter_num+1), history)
            plt.grid()
            plt.savefig(get_path('evol_history.png'))

        return history, self.population[self.idx_best_solution]

    def fitness_func(self, chrom):
        return sum(bit for bit in chrom)

    def calculate_fitnesses(self, pop):
        return tuple(self.fitness_func(c) for c in pop)

"""Genetic Algorithm multiproccesses implementation"""
class GA_Multiprocessing(GA):
    def __init__(self, pop_size=200, str_size=20):
        super().__init__(pop_size, str_size)
        self.pc = cpu_count()
        self.chunk_size = int(np.ceil(self.population.shape[0]/self.pc))

    def calculate_fitnesses(self, pop):
        pool = Pool(self.pc)
        results = [pool.apply_async(self.async_func, args=(pop[i*self.chunk_size:(
            i*self.chunk_size+self.chunk_size-(pop.shape[0]%self.pc if i+1==self.pc else 0))],))
            for i in range(self.pc)]
        pool.close()
        pool.join()

        return np.concatenate([r.get() for r in results])

    def async_func(self, pop_chunk):
        return [self.fitness_func(chrom) for chrom in pop_chunk]


"""Genetic Algorithm continuous implementation"""
class CGA(GA):
    def __init__(self, pop_size=100, l_bound=-10, u_bound=10, gene_size=10, var_num=2):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.gene_size = gene_size
        self.var_num = var_num
        super().__init__(pop_size, gene_size*var_num)

    def fitness_func(self, chrom):
        return sum(self.decode(chrom[i*self.gene_size:i*self.gene_size+self.gene_size])**2
            for i in range(self.var_num))

    def decode(self, gene):
        return self.l_bound + ((self.u_bound-self.l_bound)/(2**self.gene_size-1)
            )*sum(2**i*b for i, b in enumerate(gene))


"""Genetic Algorithm implementation for optimize weights and delays in a neural network"""
class NNGA(GA_Multiprocessing):
    def __init__(self, snn, dataset, c_size=10):
        self.snn = snn
        self.dataset = dataset
        self.c_size = c_size # chromosome size
        self.sys_size = len(snn.I)*len(snn.H)+len(snn.H)
        self.half_size = self.sys_size*c_size
        super().__init__(200, self.half_size*2)

    def run(self, iter_num=50, graph=False):
        _, best_chromosome = super().run(iter_num, graph)
        return self.decode_chrom(best_chromosome)

    def fitness_func(self, chrom):
        self.snn.set_neural_synapses(self.decode_chrom(chrom))

        avg_ftimes = []
        for cl in self.dataset:
            ftimes = []
            for e in cl:
                ftime = self.snn.simulate(e)
                if ftime == None:
                    return maxsize
                ftimes.append(ftime)
            avg_ftimes.append(np.mean(ftimes))

        success = 0
        for cl_idx, cl in enumerate(self.dataset):
            for e in cl:
                ftime = self.snn.simulate(e)
                argmin = np.argmin(tuple(abs(avg_ftime-ftime) for avg_ftime in avg_ftimes))
                if cl_idx == argmin:
                    success += 1

        return 1-success/sum(cl.shape[0] for cl in self.dataset)

    def decode_chrom(self, chrom):
        decode_weight = lambda gene: 10/(2**self.c_size-1)*sum(2**i*b for i, b in enumerate(gene))
        decode_delay = lambda gene: 1/(2**self.c_size-1)*sum(2**i*b for i, b in enumerate(gene))

        return tuple((decode_weight(chrom[ci:ci+self.c_size]),
                decode_delay(chrom[self.half_size+ci:self.half_size+ci+self.c_size]))
                for ci in (sys_index*self.c_size for sys_index in range(self.sys_size)))


def main():
    import pandas as pd
    from snn import SNN
    from utils import get_divs, get_tr_te_subsets

    dataset = pd.read_csv('datasets/iris.data', header=None).values
    cl_col = 4
    fe_cols = (1, dataset.shape[1]) if cl_col == 0 else (0, dataset.shape[1]-1)
    divs = get_divs(dataset, cl_col)

    # ***** Setting up training and testing subsets *****
    tr_features, _, _, _ = get_tr_te_subsets(dataset[:, fe_cols[0]:fe_cols[1]], divs)

    del divs

    # ****** Definition of neural network ******
    fe_ranges = tuple((col.min(), col.max()) for col in (dataset[:, j] for j in range(*fe_cols)))
    snn = SNN(fe_ranges)

    NNGA(snn, tr_features).run(400, True)


if __name__ == '__main__':
    main()
