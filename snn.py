import numpy as np


"""Neuron base class"""
class Neuron():
    firing_time = None
    active = True

    """Calculate if the neuron has fired in a given time, and return its value if has"""
    def compute_ftime(self, t):
        raise NotImplementedError

"""Spike Response Model based Neuron"""
class SRM_Neuron(Neuron):
    def __init__(self):
        self.tau = 10
        self.theta = 4
        self.pre_syns = [] # (neuron, synaptic weight, delay)

    def compute_ftime(self, t):
        if self.firing_time == None:
            neuron_state = self.get_neuron_state(t)
            if neuron_state[0] > self.theta:
                self.firing_time = t
                return t
            if neuron_state[1]:
                self.active = False

        return self.firing_time

    """Get membrane potential and active/inactive status of current neuron"""
    def get_neuron_state(self, t):
        # turn neuron as inactive if all its pre-synaptic neurons are inactive
        pre_syns = tuple((filter(lambda syn: syn[0].active, self.pre_syns)))
        if len(pre_syns) == 0:
            self.active = False
            return (0, False)

        # compute neuron current membrane potential and pre-synaptic spikes effect on it
        summation = 0
        lost_effect = True # Have all spikes stopped contributing to the mem. potential summation?
        for syn in pre_syns:
            if syn[0].compute_ftime(t) != None:
                te = t - syn[0].firing_time - syn[2] # time elapsed since arrival
                if te > 0:
                    summation += syn[1] * (te*np.exp(1-te/self.tau)/self.tau)
                    # if te is higher than tau then spike contribution starts to decrease
                    if te <= self.tau:
                        lost_effect = False
            else:
                lost_effect = False

        return summation, lost_effect

"""Guassian Receptive Field Neuron"""
class GRF_Neuron(Neuron):
    def __init__(self, m, min_val, max_val, i):
        self.beta = 2
        self.center = min_val + ((2*i-3)/2)*((max_val-min_val)/(m-2))
        self.sd = (1/self.beta)*((max_val-min_val)/(m-2))

    def compute_ftime(self, t):
        if t < self.firing_time:
            return None

        return self.firing_time

    """Converts a continuous input value into a firing time"""
    def encode(self, val):
        self.firing_time = np.around((1-np.exp(-(val-self.center)**2/(2*self.sd**2)))*10)

"""Spiking neural network"""
class SNN():
    def __init__(self, fe_ranges, m=4, h_lsize=10):
        self.m = m # number of encoding neurons per feature in input layer
        self.h_lsize = h_lsize # number of neurons in hidden layer

        # set up neuron layers
        self.I = tuple(GRF_Neuron(m, r[0], r[1], i) for r in fe_ranges for i in range(1, m+1))
        self.H = tuple(SRM_Neuron() for _ in range(h_lsize))
        self.O = SRM_Neuron()

    """Establish synapses between neurons (all vs all) using the given weights and delays"""
    def set_neural_synapses(self, wd):
        for i, n in enumerate(self.H):
            n.pre_syns = [(hi_n, *wd[j*self.h_lsize+i]) for j, hi_n in enumerate(self.I)]

        self.O.pre_syns = [(hi_n, *wd[self.h_lsize*len(self.I)+i]) for i, hi_n in enumerate(self.H)]

    """Simulate neural network dynamics with the given sample and return firing time of
    the output layer neuron if any"""
    def simulate(self, sample):
        # encode sample variables into firing times
        for i_idx, i in enumerate(self.I):
            i.encode(sample[int(i_idx/self.m)])

        t = 1
        while self.O.compute_ftime(t) == None and self.O.active:
            t += 1
        firing_time = self.O.firing_time

        # reset neuron firing times and active/inactive states
        for i in self.I:
            i.firing_time = None

        for h in self.H:
            h.firing_time = None
            h.active = True

        self.O.firing_time = None
        self.O.active = True

        return firing_time


def main():
    import pandas as pd

    dataset, cl_col = pd.read_csv('datasets/iris.data', header=None).values, 4
    fe_cols = (1, dataset.shape[1]) if cl_col == 0 else (0, dataset.shape[1]-1)

    fe_ranges = tuple((col.min(), col.max()) for col in (dataset[:, j] for j in range(*fe_cols)))
    SNN(fe_ranges)

if __name__ == '__main__':
    main()
