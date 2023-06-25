import torch
import gpytorch
from scipy.stats import norm
from gpytorch.kernels import RBFKernel
from torchmin import minimize_constr
from tqdm import tqdm
from copy import deepcopy


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, l0):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = RBFKernel(active_dims=tuple(range(4)),has_lengthscale = True, ard_num_dims=4)
        if not l0:
            kernel = kernel * RBFKernel(active_dims=tuple([4]), has_lengthscale = True,ard_num_dims=1) * RBFKernel(active_dims=tuple([5]),has_lengthscale = True,ard_num_dims=1)
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BayesianOptimization:
    def __init__(self, initial_sample, objective, bounds, cost, n_fidel,acq_func):
        # initialize likelihood and model
        self.device = "cpu"
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.l0_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.ln_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.training_iter = 50
        if acq_func == "ei":
            self.acquisition_function = self.ei_acq
        elif acq_func == "takg":
            self.acquisition_function = self.takg
        elif acq_func == "kg":
            self.acquisition_function = self.kg
        self.objective = objective
        self.bounds = bounds.to(self.device)
        self.x = self.generate_x(initial_sample).to(self.device)
        self.n_fidel = n_fidel
        self.s = torch.ones(n_fidel).to(self.device)
        self.x = self.x.float()
        self.starting_x = self.x[0].unsqueeze(0)
        self.cost = cost
        # Real Values
        self.g = [objective(x_i, self.bounds) for i, x_i in enumerate(torch.unbind(self.x, dim=0), 0)]
        # Noisy observations
        self.g = torch.tensor(self.g,device=self.device)
        self.g = self.g.float()
        self.y = self.g
        self.model = ExactGPModel(self.x, self.y, self.likelihood, False).to(self.device)
        self.l0_model = ExactGPModel(self.x[:, :-n_fidel], self.y, self.l0_likelihood, True).to(self.device)
        self.ln_model =ExactGPModel(self.x, self.y, self.ln_likelihood, False).to(self.device)
        self.max_mean = 0.0
        self.best_y = torch.argmin(self.y)
        self.best_x = self.x[self.best_y]
        self.fit_model(self.model, self.likelihood, self.x, self.y)
        self.fit_model(self.l0_model, self.l0_likelihood, self.x[:, :-n_fidel], self.y)
        self.fit_model(self.ln_model, self.ln_likelihood, self.x, self.y)

    def compute_posterior(self,x, model, likelihood):
        model.eval()
        likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(x))
        return observed_pred

    def ei_acq(self):
        # Choose point with maximum acquisition function value
        x = self.generate_x(1)
        bounds = {"lb": torch.zeros(self.bounds[:, 0].shape), "ub": torch.ones(self.bounds[:, 1].shape)}
        x_next = minimize_constr(self.ei, x, bounds=bounds)['x']  # x_values[torch.argmax(y_values)]
        return x_next

    def ei(self, x_values):
        # Define acquisition function
        def max0(f):
            return max(f, 0)

        # Compute acquisition function for each point in search space
        posterior = self.compute_posterior(x_values, self.model, self.likelihood)
        mean = posterior.mean
        std = posterior.stddev
        if mean.max() > self.max_mean:
            self.max_mean = mean.max()
        improv = (mean - self.max_mean)
        z = improv / std
        improv_plus = torch.zeros(improv.shape, device=self.device)
        for i in range(len(improv)):
            improv_plus[i] = max0(improv[i])
        pdf_z = torch.tensor(norm.pdf(z.detach().numpy()), device=self.device)
        cdf_z = torch.tensor(norm.cdf(z.detach().numpy()), device=self.device)
        y_values = improv_plus + std * pdf_z - torch.abs(improv) * cdf_z
        return -y_values

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

        # Simulate y_n+1 and get current mean
        mu_n1 = self.compute_mean(x)
        mu_n1 = torch.tensor(list(map(lambda x: x if x > self.max_mean else self.max_mean, mu_n1)))
        kg = (mu_n1 - self.max_mean)

        return kg.mean()

    def sample_gradient(self, x1):
        x1.requires_grad_(True)
        y = self.sample_posterior(x1)
        a = torch.autograd.grad(outputs=y, inputs=x1, allow_unused=True)
        return a[0]

    def sample_l0_gradient(self, x1):
        x1.requires_grad_(True)
        mean = self.sample_l0_posterior(x1)
        a = torch.autograd.grad(outputs=mean, inputs=x1, allow_unused=True)
        return a[0]

    # Algorithm 4 Simulation of unbiased stochastic gradient
    def kg_gradient(self, x, J=2):
        Gs = []
        for j in range(1, J):
            y1 = self.sample_posterior(x)

            x1 = self.infer_x1(y1)

            G = self.sample_gradient(x1)
            # Unpack result
            Gs.append([G[0][0], G[0][1], G[0][2]])
        Gs = torch.tensor(Gs, device=self.device)
        return Gs.mean()

    def get_x_from_kg(self, R=10, T = 3, a=4):
        kgs = []
        xs = []
        for r in range(1, R):
            x_r = self.generate_x(1)
            x = [x_r]
            for t in range(1, T):
                # Get the mean and variance of the derivative
                G = self.kg_gradient(x[t - 1], 5)
                alpha_t = a / (a + t)
                # Gradient ascent
                x_t = x[t - 1] + alpha_t * G
                x.append(x_t)
            kgs.append(self.kg_computation(x_t))
            xs.append(x_t)
        # Return x_t with largest KG
        kgs_t = torch.tensor(kgs, device=self.device)
        res = xs[torch.argmax(kgs_t)][0]
        return res.unsqueeze(0)


    def compute_sigma_tilde_with_gradient(self, xs, x1):
        term1 = self.ln_model.covar_module(x1, xs).to_dense()
        auto_covar = self.ln_model.covar_module(xs, xs).to_dense()
        #auto_covar + variance where variance is an hyperparameter
        variance = 1
        chol_covar = torch.linalg.cholesky(auto_covar)

        term2 = torch.cholesky_inverse(chol_covar)
        w = torch.randn(term1.shape).T.to(self.device)
        res = term1 @ term2 @ w
        return res

    def compute_grad_Ln(self, x):
        xs = deepcopy(self.x)
        x = deepcopy(x)
        x1, f_min = self.compute_min_x1_posterior(x)
        x1 = torch.repeat_interleave(x1, xs.shape[0],dim=0)
        xs.requires_grad_(True)
        sigma_tilde = self.compute_sigma_tilde_with_gradient(xs, x1)
        sigma_tilde.backward(gradient = torch.ones((xs.shape[0], xs.shape[0]),device=self.device))
        grad_Ln = xs.grad
        return xs, grad_Ln

    def compute_grad_L0(self, x):
        x = deepcopy(x)
        x1 = self.compute_min_x1_objective(x)
        return self.sample_l0_gradient(x1)

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

    def sample_l0_posterior(self, x):
        self.l0_model.eval()
        self.l0_likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = self.l0_likelihood(self.l0_model(x[:, :-self.n_fidel]))
        return observed_pred.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(0)


    def compute_min_x1_objective(self, x_star):
        x = deepcopy(x_star)
        x.requires_grad_(True)
        bounds = self.bounds
        bounds = {"lb": torch.zeros(bounds[:, 0].shape), "ub": torch.ones(bounds[:, 1].shape)}
        bounds['lb'][-2:] = 1.0
        opt_result = minimize_constr(self.sample_l0_posterior, x, bounds=bounds)

        return opt_result.x

    # Computation of dLn = En[g(x',1) | y(x,S)]
    def compute_dLn(self, x1):

        mean = self.sample_ln_posterior(x1)
        sigma_tilde = self.compute_sigma_tilde_with_gradient(self.x, x1)
        dLn = mean + sigma_tilde
        return dLn

    # Unbiased estimate of Ln by simulation
    def compute_Ln(self, x, R = 5):
        Ln = torch.zeros(R)
        x_r = deepcopy(x)
        temp_x = self.x
        temp_y = self.y

        for r in range(R):
            g = self.sample_posterior_batch(x_r, x_r.shape[0], self.model, self.likelihood)
            y = g
            temp_x = torch.cat([temp_x, x_r])
            temp_y = torch.cat([temp_y, y])
            self.model.set_train_data(temp_x, temp_y, strict=False)
            # Compute min x' and f_min
            x_min, f_min = self.compute_min_x1_posterior(x_r)
            Ln[r] = f_min
        return Ln.mean()

    #Input shape : d
    def compute_L0(self, x, R = 5):
        L0 = torch.zeros(R)
        x = deepcopy(x)

        for r in range(R):
            bounds = self.bounds
            bounds = {"lb": torch.zeros(bounds[:, 0].shape), "ub": torch.ones(bounds[:, 1].shape)}
            opt_result = minimize_constr(self.sample_l0_posterior, x, bounds=bounds)
            L0[r] = opt_result.fun
        return L0.mean()

    def takg_gradient(self, x1):
        # Gradient ascent
        xs, G = self.compute_grad_Ln(x1)
        x1 = x1.detach()
        G0 = self.compute_grad_L0(x1)

        # Gradient of Ln by linearity
        gradLn = G0 - G
        return gradLn

    def correct_bounds(self, x):
        for i in range(len(x[0])):
            if x[0][i] <= 0:
                x[0][i] = 0.001
            elif x[0][i] >= 1:
                x[0][i] = 0.99
        x[0][-2:] = self.bounds[-2:, 0]
        x[0][-1] = self.bounds[-1, 0] / 10000
        return x

    def fid1(self, x):
        x1 = deepcopy(x)
        for b in range(x1.shape[0]):
            x[b,-self.n_fidel:] = 1
        return x1


    def takg(self, R = 10):

        takgs = []
        xkg = []
        x_star = self.generate_x(1)
        temp_y = self.y
        temp_x = self.x
        for _ in tqdm(range(R), desc=f'takg: '):

            L0 = self.compute_L0(x_star)
            Ln = self.compute_Ln(x_star)
            #Compute cost
            cost = self.cost(x_star.squeeze(0))
            takgs.append((L0 - Ln) / cost)
            gradLn = self.takg_gradient(x_star).mean(axis=0)
            gradLn[-self.n_fidel:] = 0
            x_star = x_star + 0.01 * gradLn
            xkg.append(x_star)
            y = self.sample_posterior_batch(x_star, x_star.shape[0], self.ln_model, self.ln_likelihood)
            temp_x = torch.cat((temp_x, x_star))
            y_next = y
            temp_y = torch.cat((temp_y, y_next))
            self.ln_model.set_train_data(temp_x, temp_y, strict=False)

        kgs_t = torch.tensor(takgs, device=self.device)
        res = xkg[torch.argmax(kgs_t)]
        return res

    # Generate x constrained by self.bounds and of dimension sample_size
    def generate_x(self, sample_size):
        unif = torch.rand(size=(sample_size, len(self.bounds)), device=self.device) #* self.bounds[:, 1] + self.bounds[:, 0]
        return unif

    # Optimizes Gaussian Process parameters
    def fit_model(self, model, likelihood, train_x, train_y):
        # Find optimal model parameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # Loss for GPs is the the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = model(train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()

            # Step
            optimizer.step()

    def sample_posterior(self, x):
        self.model.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
        return observed_pred.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(0)

    def sample_ln_posterior(self, x):
        self.ln_model.eval()
        self.ln_likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = self.ln_likelihood(self.ln_model(x))
        return observed_pred.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(0)

    # Predicts
    def sample_posterior_batch(self, x, sample_size, model, likelihood):
        model.eval()
        likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(x))

        return observed_pred.rsample(sample_shape=torch.Size([sample_size])).squeeze(0)

    # Optimizes objective function
    def optimize(self, num_iter):
        c = 0.0
        for _ in tqdm(range(num_iter)):
            # Acquire most likely X
            x_next = self.acquisition_function()
            x_next = x_next.float().detach()

            g_nexts = torch.zeros(x_next.shape[0])

            # Heavy function call
            for i in range(x_next.shape[0]):
                g_nexts[i] = self.objective(x_next[i], self.bounds)
                c += self.cost(x_next[i])
            g_nexts = g_nexts.float().to(self.device)

            # Add new data points to X and Y
            self.x = torch.vstack((self.x, x_next))
            self.g = torch.cat((self.g, g_nexts))
            y_next = g_nexts
            self.y = torch.cat((self.y, y_next))
            self.model.set_train_data(self.x, self.y, strict=False)
            self.l0_model.set_train_data(self.x[:, :-self.n_fidel], self.y, strict=False)
            self.ln_model.set_train_data(self.x, self.y, strict=False)
        best_idx = torch.argmin(self.y)
        self.best_x = self.x[best_idx]
        self.best_y = self.y[best_idx]
        return self.best_x, self.best_y, c, self.x, self.y
