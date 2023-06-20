import torch
from FCN import FCN_OPT

class MLP_benchmark:
  def __init__(self):
    lr_bounds = [10 ** (-6), 1]
    dropout_rate_bounds = [0, 1]
    batch_size_bounds = [32, 256]
    n_units_bounds = [100, 1000]
    epochs = [0.6, 0.6]
    train_size = [5500, 5500]
    self.bounds = torch.tensor([lr_bounds, dropout_rate_bounds, batch_size_bounds, n_units_bounds, epochs, train_size])

  def objective(self, x):
    hyperparams = x #* (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    hyperparams = hyperparams.tolist()
    print(*hyperparams)
    net = FCN_OPT(*hyperparams, val_size=5000, device='cuda:0')
    net.train()
    val_err = net.evaluate()
    print(f"val_err: {val_err}")
    return val_err


class Branin_benchmark:

  def __init__(self):
    self.bounds = torch.tensor([[-5, 10], [0, 15], [0,1]])
    self.x_obj = (torch.tensor([-3.14, 12.28]), torch.tensor([3.14, 2.28]), torch.tensor([9.42, 2.48]))
    self.y_obj = torch.tensor(0.397887)

  def objective(self, x):
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

