import torch
from FCN import FCN_OPT

class MLP_benchmark:
  def __init__(self, n_fidel):
    lr_bounds = [-6, 0]
    dropout_rate_bounds = [0, 1]
    batch_size_bounds = [32, 256]
    n_units_bounds = [100, 1000]
    epochs = [10, 10]
    train_size = [1000, 5500]
    self.bounds = torch.tensor([lr_bounds, dropout_rate_bounds, batch_size_bounds, n_units_bounds, epochs, train_size])
    self.n_fidel = n_fidel

  def objective(self, x, bounds):
    hyperparams = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    hyperparams = hyperparams.tolist()
    hyperparams[0] = 10**hyperparams[0]
    print(*hyperparams)
    net = FCN_OPT(*hyperparams, val_size=5000, device='cuda:0')
    net.train()
    val_err = net.evaluate()
    print(f"val_err: {val_err}")
    return val_err

  def cost(self, x):
    epochs = float(self.bounds[-2][1])
    if self.n_fidel == 2:
        epochs *= x[-2]
    train_size = (x[-1] * self.bounds[-1][1] + self.bounds[-1][0])
    cost = (epochs * train_size) / ((self.bounds[-1][1]) * self.bounds[-2][1])
    return cost

class Branin_benchmark:

  def __init__(self):
    self.bounds = torch.tensor([[-5, 10], [0, 15], [0,1]])
    self.x_obj = (torch.tensor([-3.14, 12.28]), torch.tensor([3.14, 2.28]), torch.tensor([9.42, 2.48]))
    self.y_obj = torch.tensor(0.397887)

  def objective(self, x, bounds):
    x1 = x[0]
    x2 = x[1]
    s1 = x[2]
    a = 1
    b = 5.1 / ((2 * torch.pi) ** 2) - 0.1 * (1 - s1)
    c = 5 / torch.pi
    r = 6
    s = 10
    t = 1 / (8 * torch.pi)
    term1 = a * (torch.square(x2 - b * torch.square(x1) + c*x1 - r))
    term2 = s*(1-t)*torch.cos(x1)
    f = term1 + term2 + s
    return f

