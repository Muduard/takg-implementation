import os
from matplotlib import pyplot as plt
import time
from bo import BayesianOptimization
from synt_funcs import MLP_benchmark, Branin_benchmark



def evaluate_MLP():
    #Hyperparameters
    initial_sample = 1

    benchmark = Branin_benchmark()
    costs = []
    ys = []
    for i in range(1):
        bo = BayesianOptimization(initial_sample, benchmark.objective, benchmark.bounds)
        bo.fit_model()
        x, y = bo.optimize(5)
        ys.append(y.detach().numpy())
        print(f"ys: {ys}")
        print(x.detach().numpy())
        epochs = x[-2] * 20
        train_size = (x[-1] * benchmark.bounds[-1][1] + benchmark.bounds[-1][0])
        cost = (epochs * train_size) / (20 * benchmark.bounds[-1][1])
        cost = cost.detach().numpy()
        costs.append(cost)

    plt.plot(costs, ys)
    plt.show()


evaluate_MLP()

