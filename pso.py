from multiprocessing import Pool, cpu_count
# from sys import maxsize

import numpy as np
import matplotlib.pyplot as plt

from utils import get_path


"""Particle Swarm Optimization Algorithm"""
class PSO():
    def __init__(self, dataset, sn_model, save_plot=False, **kwargs):
        # Defining initial constants
        self.dataset = dataset # n dimentional numpy array
        self.fe_size = dataset[0].shape[1]
        self.iters = 50 # TODO set iterations number as run method's parameter
        self.max_vel = .2
        self.pop = 50
        self.c1, self.c2 = .0205, .0205
        self.save_plot = save_plot

        # SN Model
        self.sn_model = sn_model

        # Update instance attributes with passed kwargs
        self.__dict__.update(kwargs)

        # Definition of the initial random values and the initial best set
        self.swarm = np.random.rand(self.pop, self.fe_size)
        self.velocities = np.random.rand(self.pop, self.fe_size)
        # for i in range(self.pop):
        #     for j in range(0 if i%2 == 0 else 1, self.fe_size, 2):
        #         self.swarm[i,j], self.velocities[i,j] = -self.swarm[i,j], -self.velocities[i,j]
        self.sw_best = self.swarm.copy()
        self.sw_best_fitnesses = np.full(self.pop, np.inf)
        self.global_idx = 0

        # Array to track the evolution of the algorithm
        self.history = np.empty(self.iters, dtype=np.float64) # TODO put history inside run method

    """Run the evolutive algorithm"""
    def run(self):
        for iteration in range(self.iters):
            # Compare actual swarm fitnesses vs best ones
            self.update_best_particles()

            # Update best global
            self.global_idx = np.argmin(self.sw_best_fitnesses)

            # Update velocity and position for each particle
            for i, vel in enumerate(self.velocities):
                self.velocities[i] = vel + np.random.rand(self.fe_size)*self.c1*(
                        self.sw_best[i]-self.swarm[i]) + np.random.rand(self.fe_size)*self.c2*(
                        self.sw_best[self.global_idx]-self.swarm[i])

                for j, vel_dim in enumerate(self.velocities[i]):
                    if abs(vel_dim) > self.max_vel:
                        vel[j] = self.max_vel * vel_dim/abs(vel_dim)

                self.swarm[i] = self.swarm[i] + self.velocities[i]

            self.history[iteration] = self.sw_best_fitnesses[self.global_idx]
            print(f'iteration {iteration+1}: {self.sw_best_fitnesses[self.global_idx]}')

        if self.save_plot:
            plt.plot(range(self.iters), self.history)
            plt.savefig(get_path('w_evolution_history.png'))

        return self.sw_best[self.global_idx], self.history

    """Compare fitnesses of the actual swarm population vs historically best ones"""
    def update_best_particles(self):
        for i, p in enumerate(self.swarm):
            p_fitness = self.fit_func(p)
            if p_fitness < self.sw_best_fitnesses[i]:
                self.sw_best_fitnesses[i] = p_fitness
                self.sw_best[i] = p

        return

    """Objective function of the evolutive process"""
    def fit_func(self, vals):
        afr = np.empty(len(self.dataset), dtype=np.float64)
        # sdfr = np.empty(len(self.dataset), dtype=np.float64)

        # Get the average firing rates per class
        # For each class
        for cl_idx, cl in enumerate(self.dataset):
            fi_rates = np.empty(cl.shape[0], dtype=np.float64)

            # For each element
            for i, e in enumerate(cl):
                fi_rates[i] = self.sn_model.run(np.dot(e, vals))

            afr[cl_idx] = np.mean(fi_rates)
            # sdfr[cl_idx] = np.std(fi_rates)

        # Compute accuracy with the obtained afrs
        success = 0
        # For each class
        for cl_idx, cl in enumerate(self.dataset):
            # For each element
            for i, e in enumerate(cl):
                m_idx = np.argmin(tuple(abs(self.sn_model.run(np.dot(e, vals))-fr) for fr in afr))

                if m_idx == cl_idx:
                    success += 1

        # 1 - acc
        return 1 - success/sum(tuple(cl.shape[0] for cl in self.dataset))
        # # Calculate dist(AFR)
        # dis_afr = 0.0
        # for i in range(afr.shape[0]-1):
        #     dis_afr += abs(afr[i+1]-afr[i])

        # return maxsize if dis_afr == 0 else 1/dis_afr + sum(sdfr)

"""PSO Multiprocessing implementation"""
class PSOMultiprocessing(PSO):
    def __init__(self, dataset, sn_model, save_plot=False):
        super().__init__(dataset, sn_model, save_plot, pop=200, iters=10)
        self.pc = cpu_count()

    def update_best_particles(self):
        # Configuring and executing pool
        pool = Pool(self.pc)
        results = [pool.apply_async(self.prll_fitness, args=(i,)) for i in range(self.pc)]
        pool.close()
        pool.join()

        # Compare actual swarm fitnesses vs best ones
        for result in (r.get() for r in results):
            for i, p_fitness in result:
                if p_fitness < self.sw_best_fitnesses[i]:
                    self.sw_best_fitnesses[i] = p_fitness
                    self.sw_best[i] = self.swarm[i]

    """Function excecuted by each process in the pool"""
    def prll_fitness(self, ini):
        return [(i, self.fit_func(self.swarm[i])) for i in range(ini, self.pop, self.pc)]


def main():
    import pandas as pd
    from utils import get_divs, norm_features, get_tr_te_subsets
    from bms import BMS
    from srm import SRM

    path, cl_col = 'datasets/iris.data', 4

    dataset = pd.read_csv(path, header=None).values
    fe_cols = (1, dataset.shape[1]) if cl_col == 0 else (0, dataset.shape[1]-1)
    divs = get_divs(dataset, cl_col)
    tr_features, _ = get_tr_te_subsets(dataset[:, fe_cols[0]:fe_cols[1]], divs)

    # sn_model = BMS()
    sn_model = SRM()

    best, history = PSOMultiprocessing(tr_features, sn_model).run()
    print('[', ', '.join(map(str, best)), ']')
    plt.plot(range(history.shape[0]), history, c='b')
    plt.show()


if __name__ == "__main__":
    main()
