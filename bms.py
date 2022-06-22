import numpy as np


class BMS():
    # Optimal parameters: gamma=.68, theta=7
    def __init__(self, gamma=0.68, theta=7):
        self.gamma = gamma
        self.theta = theta

    def run(self, i_ext):
        fr = -1
        vt = 0
        for _ in range(100):
            vt = self.gamma*vt*(1-(0 if vt < self.theta else 1)) + i_ext
            if vt == i_ext:
                fr += 1

        return fr/100

    def get_firing_trace(self, i_ext):
        fire_trace = []
        vt = 0
        for t in range(100):
            vt = self.gamma*vt*(1-(0 if vt < self.theta else 1)) + i_ext
            if vt == i_ext:
                fire_trace.append(t)

        return np.array(fire_trace[1:]), (len(fire_trace)-1)/100

    def get_voltage_trace(self, i_ext):
        vt_trace = np.arange(100, dtype=np.float64)

        vt = 0
        for t in range(100):
            vt_trace[t] = vt
            vt = self.gamma*vt*(1-(0 if vt < self.theta else 1)) + i_ext

        return vt_trace


# i_ext > (1-gamma)*theta
def main():
    import numpy as np

    ini, end, step = 5, 25, 1
    frs = np.empty(int((end-ini)/step), dtype=np.float64)
    for i, i_ext in enumerate(np.arange(ini, end, step, dtype=np.float64)):
        frs[i] = BMS().run(i_ext)
        print(f'i_ext({i_ext})={frs[i]}')
    print(np.std(frs))


def graph_bms():
    import matplotlib.pyplot as plt
    import numpy as np

    v = BMS().get_voltage_trace(3.1)
    t = np.arange(1, v.shape[0]+1)

    plt.plot(t, v)
    plt.show()


if __name__ == '__main__':
    main()
    # graph_bms()
