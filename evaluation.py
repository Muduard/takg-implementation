import os

import numpy as np
from matplotlib import pyplot as plt
import time
from bo import BayesianOptimization
from synt_funcs import MLP_benchmark, Branin_benchmark
from copy import deepcopy
import torch

def evaluate_MLP():
    #Hyperparameters
    initial_sample = 1
    n_fidel = 1
    benchmark = MLP_benchmark(n_fidel)
    jys = []
    jxs = []
    jcosts = []
    xs = []
    ys = []
    s = np.linspace(0.9, 0.9, 1)
    with open("ress.txt", "a") as f:
        f.write("Starting benchmark\n")
        for j in range(1):
            best_ys = []
            best_xs = []
            costs = []
            for i in range(1):

                current_bounds = deepcopy(benchmark.bounds).float()
                current_bounds[-n_fidel:, :] = torch.tensor(s[i]).repeat(n_fidel, 2)
                print(current_bounds)
                bo = BayesianOptimization(initial_sample, benchmark.objective, current_bounds, benchmark.cost, n_fidel, "takg")
                x_best, y_best, cost, x, y = bo.optimize(3)
                best_ys.append(y_best.detach().numpy())
                best_xs.append(x_best.detach().numpy())
                xs.append(x.detach().numpy())
                ys.append(y.detach().numpy())
                f.write(f"({i},{j}):\n")
                f.write("X: ")
                f.write(str(best_xs))
                f.write("\nY: ")
                f.write(str(best_ys))
                f.write("\nCost: ")
                f.write(str(cost))
                f.write("\n------------------------------------------\n")
                print(f"ys: {best_ys}")
                print(f"xs:{x_best}")
                print(f"cost:{cost}")
                costs.append(cost)
            jys.append(best_ys)
            jxs.append(best_xs)
            jcosts.append(costs)

    jcosts = np.array(jcosts).mean(axis=0)
    jys = np.array(jys).mean(axis=0) * 100
    Z = [x for _, x in sorted(zip(jcosts, jys))]
    jcosts = sorted(jcosts)
    plt.plot(jcosts, Z)
    plt.show()


evaluate_MLP()

