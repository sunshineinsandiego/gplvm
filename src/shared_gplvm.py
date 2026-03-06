import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.priors import NormalPrior

class LatentGP(ApproximateGP):
    """
    A standard Variational GP that maps from Latent X -> Observed Y modality.
    """
    def __init__(self, n_inducing, latent_dim):
        inducing_points = torch.randn(n_inducing, latent_dim)
        variational_distribution = CholeskyVariationalDistribution(n_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class SharedVariationalGPLVM(gpytorch.Module):
    """
    Container for the Shared vGPLVM.
    Holds the shared latent parameters X and two independent GP mappings.
    """
    def __init__(self, n_data, latent_dim, n_inducing=50, Y1_dim=None, Y2_dim=None):
        super().__init__()
        
        self.n_data = n_data
        self.latent_dim = latent_dim
        
        # 2. The Prior over Latent Variables X (N(0, I))
        # This standard normal prior is what allows the model to handle missing modalities 
        # gracefully during variational imputation inference.
        self.register_parameter(
            name="X", 
            parameter=nn.Parameter(torch.randn(n_data, latent_dim))
        )
        self.register_prior("prior_X", NormalPrior(torch.zeros(1), torch.ones(1)), "X")
        
        # 3. Independent GPs for each Modality
        self.gp_img = LatentGP(n_inducing, latent_dim)
        self.gp_tab = LatentGP(n_inducing, latent_dim)
        
        # Optional: Save dimensions for downstream construction of likelihoods
        self.Y1_dim = Y1_dim
        self.Y2_dim = Y2_dim

    def forward(self, x):
        """
        Forward pass defining the Multimodal GP prior.
        Note: x is expected to be the latent space coordinates self.X.
        Returns the latent GP representation (to be mapped to Y_1 and Y_2 by likelihoods).
        """
        # Get marginal distributions from each GP
        mvn_img = self.gp_img(x)
        mvn_tab = self.gp_tab(x)
        
        # Wrap as Multitask distributions to match Y1_dim and Y2_dim
        # This effectively models D independent GPs sharing the same kernel/latent X
        mt_img = MultitaskMultivariateNormal.from_repeated_mvn(mvn_img, num_tasks=self.Y1_dim)
        mt_tab = MultitaskMultivariateNormal.from_repeated_mvn(mvn_tab, num_tasks=self.Y2_dim)
        
        return mt_img, mt_tab
               
    def sample_latent(self):
        """Helper to retrieve current latent parameters"""
        return self.X
        
    def latent_prior_loss(self):
        """
        Computes the negative log-likelihood of the latent X under the prior N(0, I).
        Minimizing this term corresponds to MAP estimation for X.
        """
        # -log p(X) ~ 0.5 * sum(X^2) (ignoring constants)
        return 0.5 * torch.sum(self.X ** 2)

def create_multimodal_likelihoods(Y1_dim, Y2_dim):
    """
    Creates standard Gaussian likelihoods for regression of continuous 
    features out of the latent space.
    """
    # Note: MultitaskGaussianLikelihood is needed because Y1 (Image Embedding) 
    # has num_tasks = D1 (e.g. 768) and Y2 has num_tasks = D2 (e.g. 5).
    likelihood_img = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y1_dim)
    likelihood_tab = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y2_dim)
    return likelihood_img, likelihood_tab
