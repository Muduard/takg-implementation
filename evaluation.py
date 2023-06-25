import numpy as np
from matplotlib import pyplot as plt
from bo import BayesianOptimization
from synt_funcs import MLP_benchmark, Branin_benchmark
from copy import deepcopy
from tqdm import tqdm

def evaluate_MLP(log_file, initial_sample = 1, n_fidel = 2, acq = "takg"):
    #Hyperparameters
    benchmark = MLP_benchmark(n_fidel)
    jys = []
    jxs = []
    jcosts = []
    xs = []
    ys = []

    with open(log_file, "a") as f:
        f.write("Starting benchmark\n")
        outer = tqdm(range(5), desc='')

        for j in outer:
            best_ys = []
            best_xs = []
            costs = []
            for i in range(5):

                current_bounds = deepcopy(benchmark.bounds).float()
                #if n_fidel != 0:
                #    current_bounds[-n_fidel:, :] = torch.tensor(s).repeat(n_fidel, 2)
                bo = BayesianOptimization(initial_sample, benchmark.objective, current_bounds, benchmark.cost, n_fidel, acq)
                x_best, y_best, c, x, y = bo.optimize(i)
                best_ys.append(y_best.detach().numpy())
                best_xs.append(x_best.detach().numpy())
                xs.append(x.detach().numpy())
                ys.append(y.detach().numpy())
                f.write(f"{i},{j} ")
                f.write(str(best_ys))
                f.write(f"\n{i},{j} ")
                f.write(str(c))
                f.write("\n")
                outer.set_description(f"i: {i},y: {max(best_ys)}, c: {c}")
                costs.append(c)
            jys.append(best_ys)
            jxs.append(best_xs)
            jcosts.append(costs)

    jcosts = np.array(jcosts)
    jys = np.array(jys)
    np.save("costs.npz", jcosts)
    np.save("ys.npz", jys)
    jcosts = jcosts.mean(axis=0) * 10
    jys = jys.mean(axis=0) * 100
    Z = [x for _, x in sorted(zip(jcosts, jys))]
    jcosts = sorted(jcosts)
    plt.plot(jcosts, Z)
    plt.show()

log_file = "result.txt"
initial_sample = 1
n_fidel = 2
acq = "takg"
evaluate_MLP(log_file, initial_sample, n_fidel, acq)

