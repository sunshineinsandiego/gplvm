import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import NormalPrior

class SharedVariationalGPLVM(ApproximateGP):
    """
    A Shared Variational Gaussian Process Latent Variable Model (vGPLVM).
    Learns a joint low-dimensional latent space X from two modalities:
    Y1 (High-dimensional Image Features) and Y2 (Low-dimensional Tabular Config).
    """
    def __init__(self, n_data, latent_dim, n_inducing=50, Y1_dim=None, Y2_dim=None):
        # 1. Variational Distribution & Strategy over Latent Space X
        inducing_points = torch.randn(n_inducing, latent_dim)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=n_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
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
        
        # 3. Covariance Kernels for each Modality
        # We assign an RBF Kernel to model smooth nonlinear mappings out of the latent space.
        self.mean_module = ZeroMean()
        
        # Image Kernel
        self.covar_module_img = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
        # Tabular Kernel
        self.covar_module_tab = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
        # Optional: Save dimensions for downstream construction of likelihoods
        self.Y1_dim = Y1_dim
        self.Y2_dim = Y2_dim

    def forward(self, x):
        """
        Forward pass defining the Multimodal GP prior.
        Note: x is expected to be the latent space coordinates self.X.
        Returns the latent GP representation (to be mapped to Y_1 and Y_2 by likelihoods).
        """
        mean_x = self.mean_module(x)
        
        # We compute two separate covariances operating on the shared coordinate space
        covar_img = self.covar_module_img(x)
        covar_tab = self.covar_module_tab(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_img), \
               gpytorch.distributions.MultivariateNormal(mean_x, covar_tab)
               
    def sample_latent(self):
        """Helper to retrieve current latent parameters"""
        return self.X

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
