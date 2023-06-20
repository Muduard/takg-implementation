import math

import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

import time
from scipy.stats import norm
from gpytorch.kernels import RBFKernel
from torchmin import minimize, minimize_constr
from tqdm import tqdm
from copy import deepcopy
from FCN import FCN_OPT

class MyMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(1)))

    def forward(self, input):
        return self.constant.expand(input.size(0), 1)

class CustomRBF(gpytorch.kernels.Kernel):
    has_lengthscale = True
    # We will register the parameter when initializing the kernel


    # this is the kernel function
    def forward(self, x1, x2, **params):
        return torch.exp(-1/2 * self.covar_dist(x1, x2))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=tuple([1]))
        kernel = RBFKernel(active_dims=tuple(range(2))) * RBFKernel(active_dims=tuple([2]))
        #kernel = RBFKernel(active_dims=tuple(range(4)), ard_num_dims=4) * RBFKernel(active_dims=tuple([4])) * \
        #         RBFKernel(active_dims=tuple([5]))
        #kernel = CustomRBF()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BayesianOptimization:
    def __init__(self, initial_sample, objective, bounds):
        # initialize likelihood and model
        self.device = "cpu"
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.training_iter = 50
        self.acquisition_function = self.takg
        self.objective = objective
        self.bounds = bounds.to(self.device)
        self.x = self.generate_x(initial_sample).to(self.device)
        self.x = self.x.float()
        self.starting_x = self.x[0].unsqueeze(0)

        # Real Values
        self.g = [objective(x_i) for i, x_i in enumerate(torch.unbind(self.x, dim=0), 0)]

        #self.g = torch.stack(, dim=0).squeeze().to(self.device)
        # Noisy observations
        self.g = torch.tensor(self.g)
        self.g = self.g.float()
        self.y = self.g
        self.model = ExactGPModel(self.x, self.g, self.likelihood).to(self.device)
        self.original_model = deepcopy(self.model)
        self.max_mean = 0.0
        self.best_y = torch.argmin(self.y)
        self.best_x = self.x[self.best_y]

    def ei(self):
        # Define acquisition function
        def max0(x):
            return max(x, 0)

        # Compute acquisition function for each point in search space
        x_values = self.generate_x(2)
        mean, std = self.predict(x_values, x_values.shape[0])

        if mean.max() > self.max_mean:
            self.max_mean = mean.max()
        improv = (mean - self.max_mean)
        z = improv / std
        improv_plus = torch.zeros(improv.shape, device=self.device)
        for i in range(len(improv)):
            improv_plus[i] = max0(improv[i])
        pdf_z = torch.tensor(norm.pdf(z.cpu()), device=self.device)
        cdf_z = torch.tensor(norm.cdf(z.cpu()), device=self.device)
        y_values = improv_plus + std * pdf_z - torch.abs(improv) * cdf_z
        # Choose point with maximum acquisition function value
        x_next = x_values[torch.argmax(y_values)]
        return x_next.unsqueeze(0)

    def kg(self):
        return self.get_x_from_kg()

    def infer_x1(self, y1):
        x = torch.randn(self.x.shape[1], device=self.device).unsqueeze(0)

        # Optimize X using the Gaussian process model
        x.requires_grad_(True)
        optimizer_x = torch.optim.LBFGS([x])
        self.model.eval()

        def closure():
            optimizer_x.zero_grad()
            output = self.model(x)
            loss = -output.log_prob(y1).sum()
            loss.backward(retain_graph=True)
            return loss
        for i in range(100):
            optimizer_x.step(closure)
        return x.detach()

    def kg_computation(self, x):

        # Simulate y_n+1 and get posteriori sample
        mu_n1 = torch.zeros(10)
        mu_n1 = self.model(x).rsample(mu_n1.shape)
        mu_n1 = torch.tensor(list(map(lambda x: x if x > self.max_mean else self.max_mean, mu_n1)))
        kg = (mu_n1 - self.max_mean)

        return kg.mean()


    def gradient(self, x1):
        x1.requires_grad_(True)
        mean = self.compute_mean(x1)
        a = torch.autograd.grad(outputs=mean, inputs=x1, allow_unused=True)
        return a[0]

    # Algorithm 4 Simulation of unbiased stochastic gradient
    def kg_gradient(self, x, J=2):
        Gs = []
        for j in range(1, J):
            mean, std = self.predict(x)
            y1 = mean + std + torch.randn(1, device=self.device)

            x1 = self.infer_x1(y1)

            G = self.gradient(x1)
            # Unpack result

            Gs.append([G[0][0], G[0][1], G[0][2]])
        Gs = torch.tensor(Gs, device=self.device)
        return Gs.mean()

    def get_x_from_kg(self, R=10, T=10 ** 2, a=4, J=10 ** 3):
        kgs = []
        T = 3
        xs = []
        for r in range(1, R):
            x_r = self.generate_x(1)
            x = [x_r]
            for t in range(1, T):
                # Get the mean and variance of the derivative
                start = time.time()
                G = self.kg_gradient(x[t - 1], 5)
                end = time.time()
                print(f'kg_gradient: {end - start}s')
                alpha_t = a / (a + t)
                # Gradient ascent
                x_t = x[t - 1] + alpha_t * G
                x.append(x_t)
            start = time.time()
            kgs.append(self.kg_computation(x_t))
            end = time.time()
            print(f'kg_computation: {end - start}s')
            xs.append(x_t)
        # Return x_t with largest KG
        kgs_t = torch.tensor(kgs, device=self.device)
        res = xs[torch.argmax(kgs_t)][0]
        return res


    def compute_x_at_all_fidelities(self, x):
        xs = []
        for s in self.S[0]:
            #Generate s of size sample size to assign to x
            ss = torch.tensor(s, device=self.device).repeat(x.shape[0]).unsqueeze(1)

            xs.append(torch.cat((x, ss), 1))

        xs = torch.cat(xs)

        return xs

    def compute_sigma_tilde_with_gradient(self, xs, x1):


        term1 = self.model.covar_module(x1, xs).to_dense()
        auto_covar = self.model.covar_module(xs, xs).to_dense()
        #auto_covar + variance where variance is an hyperparameter
        variance = 1
        chol_covar = torch.linalg.cholesky(auto_covar + variance * torch.eye(*auto_covar.shape))

        term2 = torch.cholesky_inverse(chol_covar)
        w = torch.randn(term1.shape).T
        res = term1 @ term2 @ w
        return res

    def compute_grad_Ln(self, x):
        xs = deepcopy(self.x)
        x1, f_min = self.compute_min_x1_posterior(x)
        x1 = torch.repeat_interleave(x1, xs.shape[0],dim=0)
        xs.requires_grad_(True)
        #mean, variance = self.predict(x1)
        #grad_Ln = torch.autograd.grad(variance, x1)[0]
        sigma_tilde = self.compute_sigma_tilde_with_gradient(xs,x1)
        sigma_tilde.backward(gradient = torch.ones((xs.shape[0], xs.shape[0])))
        grad_Ln = xs.grad
        return xs, grad_Ln

    def compute_grad_L0(self, x):
        x1 = self.compute_min_x1_objective(x)
        return self.gradient(x1)

    # Computes min x' of posterior returning x' and objective function min value
    def compute_min_x1_posterior(self, x):


        bounds = self.bounds
        bounds = {"lb": torch.zeros(bounds[:, 0].shape), "ub": torch.ones(bounds[:, 1].shape)}
        bounds['lb'][-2:] = 1.0
        opt_result = minimize_constr(self.compute_dLn, x, bounds=bounds)

        return opt_result.x, opt_result.fun

    def compute_mean(self, x):
        self.model.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            mean = self.model(x).mean
            return mean

    def compute_min_x1_objective(self,x_star):
        x = deepcopy(x_star)
        x.requires_grad_(True)
        bounds = self.bounds
        bounds = {"lb": torch.zeros(bounds[:, 0].shape), "ub": torch.ones(bounds[:, 1].shape)}
        bounds['lb'][-2:] = 1.0
        opt_result = minimize_constr(self.compute_mean, x, bounds=bounds)

        return opt_result.x

    # Computation of dLn = En[g(x',1) | y(x,S)]
    def compute_dLn(self, x1):

        #mean, variance = self.predict(x1)
        mean = self.compute_mean(x1)
        #xs = deepcopy(self.x)
        #x1 = torch.repeat_interleave(x1, xs.shape[0], dim=0)
        #xs.requires_grad_(True)
        sigma_tilde = self.compute_sigma_tilde_with_gradient(self.x, x1)
        dLn = mean + sigma_tilde
        return dLn

    # Unbiased estimate of Ln by simulation
    def compute_Ln(self, x, R = 10):
        Ln = torch.zeros(R)
        x_r = deepcopy(x)
        temp_x = self.x
        temp_y = self.y
        for r in range(R):
            g = self.predict(x_r, x_r.shape[0])
            # Simulate y = g(x) + eps
            y = g.squeeze(0)
            temp_x = torch.cat([temp_x, x_r])
            temp_y = torch.cat([temp_y, y])
            self.model.set_train_data(temp_x, temp_y, strict=False)
            # Compute min x' and f_min
            x_min, f_min = self.compute_min_x1_posterior(x_r)
            #print(x_min)
            #self.model = self.model.get_fantasy_model(x_min, f_min.unsqueeze(0).unsqueeze(0))
            Ln[r] = f_min
        return Ln.mean()

    #Input shape : d
    def compute_L0(self, x, R = 1):
        L0 = torch.zeros(R)
        x_r = deepcopy(x)
        for r in range(R):

            x_r[-1] = 1.0
            bounds = self.bounds
            bounds = {"lb": torch.zeros(bounds[:, 0].shape), "ub": torch.ones(bounds[:, 1].shape)}
            opt_result = minimize_constr(self.compute_mean, x_r.unsqueeze(0), bounds=bounds)
            L0[r] = opt_result.fun
        return L0.mean()

    def takg_gradient(self, x1):
        # Compute Ln
        #Ln = self.compute_Ln(x1)
        # Compute x' in E[min_x' g(x',1) | y(x,s)]
        #x_min, f_min = self.compute_min_x1_posterior(x1)
        #x0 = self.compute_min_x1_objective(x1)
        # Gradient ascent
        xs, G = self.compute_grad_Ln(x1)
        x1 = x1.detach()
        G0 = self.compute_grad_L0(x1)

        # Gradient of Ln by linearity
        gradLn = G0 - G
        return gradLn

    def correct_bounds(self,x):
        for i in range(len(x[0])):
            if x[0][i] <= 0:
                x[0][i] = 0.001
            elif x[0][i] >= 1:
                x[0][i] = 0.99
        x[0][-2:] = self.bounds[-2:, 0]
        x[0][-1] = self.bounds[-1, 0] / 10000
        return x

    def save_log(self, xs, takgs, x, takg):
        with open("res2.txt","w+") as f:
            f.write(f'x: {xs}\n')
            f.write(f'takg: {takgs}\n')
            f.write(f'Best x: {x}\n')
            f.write(f'Best takg: {takg}\n')


    def takg(self, R = 10):

        takgs = []
        xkg = []
        x_star = self.generate_x(1)
        a = 1
        temp_y = self.y
        temp_x = self.x
        for _ in tqdm(range(R), desc=f'takg: '):
            gradLn = self.takg_gradient(x_star)
            x_star = x_star + 1 * gradLn.mean(axis=0)
            #x_star = self.correct_bounds(x_star)
            L0 = self.compute_L0(x_star)
            Ln = self.compute_Ln(x_star)
            takgs.append(L0 - Ln)
            xkg.append(x_star)
            y = self.predict(x_star, x_star.shape[0])
            temp_x = torch.cat((temp_x, x_star))
            y_next = y
            temp_y = torch.cat((temp_y, y_next.squeeze(0)))

            self.model.set_train_data(temp_x, temp_y, strict=False)
            #self.fit_model()
            #self.original_model = deepcopy(self.model)
            #self.model = self.original_model

        kgs_t = torch.tensor(takgs, device=self.device)
        res = xkg[torch.argmax(kgs_t)]
        self.save_log(xkg, takgs, res, torch.max(kgs_t))
        return res

    # Generate x constrained by self.bounds and of dimension sample_size
    def generate_x(self, sample_size):
        unif = torch.rand(size=(sample_size, len(self.bounds)), device=self.device) * self.bounds[:, 1] + self.bounds[:, 0]
        return unif

    # Optimizes Gaussian Process parameters
    def fit_model(self):
        # Find optimal model parameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # Loss for GPs is the the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = self.model(self.x)

            # Calc loss and backprop gradients
            loss = -mll(output, self.y)
            loss.backward()

            # Step
            optimizer.step()

    # Predicts mean and variance of gaussian process on point x
    def predict(self, x, sample_size):
        self.model.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))

        return observed_pred.rsample(sample_shape=torch.Size([sample_size])).squeeze(0)

    # Optimizes objective function
    def optimize(self, num_iter):
        for _ in tqdm(range(num_iter)):
            # Acquire most likely X
            x_next = self.acquisition_function()
            x_next = x_next.float().detach()

            g_nexts = torch.zeros(x_next.shape[0])
            # Heavy function call
            for i in range(x_next.shape[0]):
                g_nexts[i] = self.objective(x_next[i])
            g_nexts = g_nexts.float()

            # Add new data points to X and Y
            self.x = torch.vstack((self.x, x_next))
            self.g = torch.cat((self.g, g_nexts))
            y_next = g_nexts
            self.y = torch.cat((self.y, y_next))
            print(f"y_next: {y_next}")
        best_idx = torch.argmin(self.y)
        self.best_x = self.x[best_idx]
        self.best_y = self.y[best_idx]
        return self.best_x, self.best_y
