import torch
from torch.quasirandom import SobolEngine
import numpy as np

from evotorch.core import Problem, Solution, SolutionBatch
from evotorch.tools.misc import Real, Vector
from evotorch.algorithms import CMAES
from evotorch.algorithms.searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin

from typing import Optional, Tuple, Union


def _safe_divide(a: Union[Real, torch.Tensor], b: Union[Real, torch.Tensor]) -> Union[torch.Tensor]:
    tolerance = 1e-8
    if abs(b) < tolerance:
        b = (-tolerance) if b < 0 else tolerance
    return a / b


def generate_student_t_distribution(num_samples, dim, v=10.):
    """
    Generate a batch of samples from a standard Student's t-distribution
    with v degrees of freedom. Output shape: [num_samples, dim].
    """
    # Standard normal samples
    normal_samples = torch.randn(num_samples, dim)

    # Chi-squared samples
    chi_squared_samples = torch.distributions.Chi2(v).sample((num_samples,))

    # Scale: ensure proper broadcasting
    t_samples = normal_samples / torch.sqrt(chi_squared_samples[:, None] / v)

    return t_samples


def farthest_point_sampling(data, subsample_size):
    """
    Use farthest point sampling (FPS) to subsample a batch of data in parallel.
    Input:
        data: torch.tensor, shape (N, D)
        subsample_size: int
    Output:
        subsample_idx: torch.tensor, shape (subsample_size)
    """
    N, D = data.shape
    device = data.device

    if N <= subsample_size:
        return torch.arange(N, device=device)

    dist = torch.full((N,), float("inf"), device=device)
    subsample_idx = torch.zeros(subsample_size, dtype=torch.long, device=device)

    for i in range(subsample_size):
        if i == 0:
            selected_idx = torch.randint(0, N, (1,), device=device).item()
        else:
            selected_idx = torch.argmax(dist).item()
        subsample_idx[i] = selected_idx
        selected_data = data[selected_idx].unsqueeze(0)  # shape (1, D)

        # Compute distances from selected_data to all points
        new_dist = torch.norm(data - selected_data, dim=1)
        dist = torch.minimum(dist, new_dist)

    return subsample_idx


def generate_qmc_normal(num_samples, dim):
    # Generate low-discrepancy points (Sobol sequence) in [0, 1]^dim
    sobol = SobolEngine(dimension=dim, scramble=True)
    u = sobol.draw(num_samples)
    # Transform to standard normal distribution by  Φ⁻¹(u) = sqrt(2) * erfinv(2u - 1)
    normal_samples = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2.0 * u - 1.0)
    return normal_samples


def generate_low_discrepancy_normal(num_samples, dim, ratio=0.3):
    # Determine number of samples to generate
    num_ld_samples = int(num_samples * ratio)
    num_normal_samples = num_samples - num_ld_samples

    # Generate low-discrepancy samples
    qmc_normal_samples = generate_qmc_normal(2*num_samples, dim).cuda()
    subsample_idx = farthest_point_sampling(qmc_normal_samples, num_ld_samples)
    ld_samples = qmc_normal_samples[subsample_idx]

    # Generate normal samples
    normal_samples = generate_qmc_normal(num_normal_samples, dim).cuda()

    return torch.cat((ld_samples, normal_samples), dim=0)


def generate_sigma_normal(num_samples, dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert num_samples >= 2 * dim + 1, "Number of samples must be at least 2 * dim + 1 for sigma sampling."

    # Generate 2 * dim canonical sigma points excluding the mean
    c = torch.sqrt(torch.tensor(float(dim), device=device))
    zero = torch.zeros(dim, device=device)
    eye = torch.eye(dim, device=device)
    sigma_points = torch.stack([
        c * eye,  # positive directions
        -c * eye  # negative directions
    ], dim=0).view(-1, dim)  # shape (2*dim + 1, dim)

    # Generate normal samples
    normal_samples = generate_qmc_normal(num_samples - 2 * dim, dim).cuda()

    return torch.cat((sigma_points, normal_samples), dim=0)



class CMAES_cus(CMAES):
    """
    CMA-ES algorithm for optimization.
    
    This class extends the CMA-ES algorithm from evotorch to allow custom elite size (mu).
    """
    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Real = 1.0,
        popsize: Optional[int] = 70, # NOTE. Modified to 70
        center_init: Optional[Vector] = None,
        c_m: Real = 2.0, # NOTE. Modified to 2.0
        c_sigma: Optional[Real] = None,
        c_sigma_ratio: Real = 1.0,
        damp_sigma: Optional[Real] = None,
        damp_sigma_ratio: Real = 1.0,
        c_c: Optional[Real] = None,
        c_c_ratio: Real = 1.0,
        c_1: Optional[Real] = None,
        c_1_ratio: Real = 1.0,
        c_mu: Optional[Real] = None,
        c_mu_ratio: Real = 2.0, # NOTE. Modified to 2.0
        active: bool = True,
        csa_squared: bool = False,
        stdev_min: Optional[Real] = None,
        stdev_max: Optional[Real] = None,
        separable: bool = False,
        limit_C_decomposition: bool = True,
        obj_index: Optional[int] = None,
        mu_size: Optional[int] = 10 # NOTE. Modified to 10
    ):
        """
        `__init__(...)`: Initialize the CMAES solver.

        Args:
            problem (Problem): The problem object which is being worked on.
            stdev_init (Real): Initial step-size
            popsize: Population size. Can be specified as an int,
                or can be left as None in which case the CMA-ES rule of thumb is applied:
                popsize = 4 + floor(3 log d) where d is the dimension
            center_init: Initial center point of the search distribution.
                Can be given as a Solution or as a 1-D array.
                If left as None, an initial center point is generated
                with the help of the problem object's `generate_values(...)`
                method.
            c_m (Real): Learning rate for updating the mean
                of the search distribution. By default the value is 1.

            c_sigma (Optional[Real]): Learning rate for updating the step size. If None,
                then the CMA-ES rules of thumb will be applied.
            c_sigma_ratio (Real): Multiplier on the learning rate for the step size.
                if c_sigma has been left as None, can be used to rescale the default c_sigma value.

            damp_sigma (Optional[Real]): Damping factor for updating the step size. If None,
                then the CMA-ES rules of thumb will be applied.
            damp_sigma_ratio (Real): Multiplier on the damping factor for the step size.
                if damp_sigma has been left as None, can be used to rescale the default damp_sigma value.

            c_c (Optional[Real]): Learning rate for updating the rank-1 evolution path.
                If None, then the CMA-ES rules of thumb will be applied.
            c_c_ratio (Real): Multiplier on the learning rate for the rank-1 evolution path.
                if c_c has been left as None, can be used to rescale the default c_c value.

            c_1 (Optional[Real]): Learning rate for the rank-1 update to the covariance matrix.
                If None, then the CMA-ES rules of thumb will be applied.
            c_1_ratio (Real): Multiplier on the learning rate for the rank-1 update to the covariance matrix.
                if c_1 has been left as None, can be used to rescale the default c_1 value.

            c_mu (Optional[Real]): Learning rate for the rank-mu update to the covariance matrix.
                If None, then the CMA-ES rules of thumb will be applied.
            c_mu_ratio (Real): Multiplier on the learning rate for the rank-mu update to the covariance matrix.
                if c_mu has been left as None, can be used to rescale the default c_mu value.

            active (bool): Whether to use Active CMA-ES. Defaults to True, consistent with the tutorial paper and pycma.
            csa_squared (bool): Whether to use the squared rule ("CSA_squared" in pycma) for the step-size adapation.
                This effectively corresponds to taking the natural gradient for the evolution path on the step size,
                rather than the default CMA-ES rule of thumb.

            stdev_min (Optional[Real]): Minimum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None or as a scalar.
            stdev_max (Optional[Real]): Maximum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None or as a scalar.

            separable (bool): Provide this as True if you would like the problem
                to be treated as a separable one. Treating a problem
                as separable means to adapt only the diagonal parts
                of the covariance matrix and to keep the non-diagonal
                parts 0. High dimensional problems result in large
                covariance matrices on which operating is computationally
                expensive. Therefore, for such high dimensional problems,
                setting `separable` as True might be useful.

            limit_C_decomposition (bool): Whether to limit the frequency of decomposition of the shape matrix C
                Setting this to True (default) means that C will not be decomposed every generation
                This degrades the quality of the sampling and updates, but provides a guarantee of O(d^2) time complexity.
                This option can be used with separable=True (e.g. for experimental reasons) but the performance will only degrade
                without time-complexity benefits.


            obj_index (Optional[int]): Objective index according to which evaluation
                of the solution will be done.
        """

        # Initialize the base class
        SearchAlgorithm.__init__(self, problem, center=self._get_center, stepsize=self._get_sigma)

        # Ensure that the problem is numeric
        problem.ensure_numeric()

        # CMAES can't handle problem bounds. Ensure that it is unbounded
        problem.ensure_unbounded()

        # Store the objective index
        self._obj_index = problem.normalize_obj_index(obj_index)

        # Track d = solution length for reference in initialization of hyperparameters
        d = self._problem.solution_length

        # === Initialize population ===
        self.popsize = popsize
        if not popsize:
            # Default value used in CMA-ES literature 4 + floor(3 log n)
            popsize = 4 + int(np.floor(3 * np.log(d)))
        self.popsize = int(popsize)
        # Half popsize, referred to as mu in CMA-ES literature
        self.mu = int(np.floor(popsize / 2)) 
        self._population = problem.generate_batch(popsize=popsize)

        # === Initialize search distribution ===

        self.separable = separable

        # If `center_init` is not given, generate an initial solution
        # with the help of the problem object.
        # If it is given as a Solution, then clone the solution's values
        # as a PyTorch tensor.
        # Otherwise, use the given initial solution as the starting
        # point in the search space.
        if center_init is None:
            center_init = self._problem.generate_values(1)
        elif isinstance(center_init, Solution):
            center_init = center_init.values.clone()

        # Store the center
        self.m = self._problem.make_tensor(center_init).squeeze()
        valid_shaped_m = (self.m.ndim == 1) and (len(self.m) == self._problem.solution_length)
        if not valid_shaped_m:
            raise ValueError(
                f"The initial center point was expected as a vector of length {self._problem.solution_length}."
                " However, the provided `center_init` has (or implies) a different shape."
            )

        # Store the initial step size
        self.sigma = self._problem.make_tensor(stdev_init)

        if separable:
            # Initialize C as the diagonal vector. Note that when separable, the eigendecomposition is not needed
            self.C = self._problem.make_ones(d)
            # In this case A is simply the square root of elements of C
            self.A = self._problem.make_ones(d)
        else:
            # Initialize C = AA^T all diagonal.
            self.C = self._problem.make_I(d)
            self.A = self.C.clone()

        # === Initialize raw weights ===
        # Conditioned on popsize

        # w_i = log((lambda + 1) / 2) - log(i) for i = 1 ... lambda
        self.raw_weights = self.problem.make_tensor(np.log((popsize + 1) / 2) - torch.log(torch.arange(popsize) + 1))
        # positive valued weights are the first mu
        positive_weights = self.raw_weights[: self.mu]
        negative_weights = self.raw_weights[self.mu :]

        # Variance effective selection mass of positive weights
        # Not affected by future updates to raw_weights
        self.mu_eff = torch.sum(positive_weights).pow(2.0) / torch.sum(positive_weights.pow(2.0))

        # === Initialize search parameters ===
        # Conditioned on weights

        # Store fixed information
        self.c_m = c_m
        self.active = active
        self.csa_squared = csa_squared
        self.stdev_min = stdev_min
        self.stdev_max = stdev_max

        # Learning rate for step-size adaption
        if c_sigma is None:
            c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3)
        self.c_sigma = c_sigma_ratio * c_sigma

        # Damping factor for step-size adapation
        if damp_sigma is None:
            damp_sigma = 1 + 2 * max(0, torch.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.damp_sigma = damp_sigma_ratio * damp_sigma

        # Learning rate for evolution path for rank-1 update
        if c_c is None:
            # Branches on separability
            if separable:
                c_c = (1 + (1 / d) + (self.mu_eff / d)) / (d**0.5 + (1 / d) + 2 * (self.mu_eff / d))
            else:
                c_c = (4 + self.mu_eff / d) / (d + (4 + 2 * self.mu_eff / d))
        self.c_c = c_c_ratio * c_c

        # Learning rate for rank-1 update to covariance matrix
        if c_1 is None:
            # Branches on separability
            if separable:
                c_1 = 1.0 / (d + 2.0 * np.sqrt(d) + self.mu_eff / d)
            else:
                c_1 = min(1, popsize / 6) * 2 / ((d + 1.3) ** 2.0 + self.mu_eff)
        self.c_1 = c_1_ratio * c_1

        # Learning rate for rank-mu update to covariance matrix
        if c_mu is None:
            # Branches on separability
            if separable:
                c_mu = (0.25 + self.mu_eff + (1.0 / self.mu_eff) - 2) / (d + 4 * np.sqrt(d) + (self.mu_eff / 2.0))
            else:
                c_mu = min(
                    1 - self.c_1, 2 * ((0.25 + self.mu_eff - 2 + (1 / self.mu_eff)) / ((d + 2) ** 2.0 + self.mu_eff))
                )
        self.c_mu = c_mu_ratio * c_mu

        # The 'variance aware' coefficient used for the additive component of the evolution path for sigma
        self.variance_discount_sigma = torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        # The 'variance aware' coefficient used for the additive component of the evolution path for rank-1 updates
        self.variance_discount_c = torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)

        # === Finalize weights ===
        # Conditioned on search parameters and raw weights

        # Positive weights always sum to 1
        positive_weights = positive_weights / torch.sum(positive_weights)

        if self.active:
            # Active CMA-ES: negative weights sum to alpha

            # Get the variance effective selection mass of negative weights
            mu_eff_neg = torch.sum(negative_weights).pow(2.0) / torch.sum(negative_weights.pow(2.0))

            # Alpha is the minimum of the following 3 terms
            alpha_mu = 1 + self.c_1 / self.c_mu
            alpha_mu_eff = 1 + 2 * mu_eff_neg / (self.mu_eff + 2)
            alpha_pos_def = (1 - self.c_mu - self.c_1) / (d * self.c_mu)
            alpha = min([alpha_mu, alpha_mu_eff, alpha_pos_def])

            # Rescale negative weights
            negative_weights = alpha * negative_weights / torch.sum(torch.abs(negative_weights))
        else:
            # Negative weights are simply zero
            negative_weights = torch.zeros_like(negative_weights)

        # Concatenate final weights
        self.weights = torch.cat([positive_weights, negative_weights], dim=-1)

        # === Some final setup ===

        # Initialize the evolution paths
        self.p_sigma = 0.0
        self.p_c = 0.0

        # Hansen's approximation to the expectation of ||x|| x ~ N(0, I_d).
        # Note that we could use the exact formulation with Gamma functions, but we'll retain this form for consistency
        self.unbiased_expectation = np.sqrt(d) * (1 - (1 / (4 * d)) + 1 / (21 * d**2))

        self.last_ex = None

        # How often to decompose C
        self.limit_C_decomposition = limit_C_decomposition
        if limit_C_decomposition:
            self.decompose_C_freq = max(1, int(np.floor(_safe_divide(1, 10 * d * (self.c_1.cpu() + self.c_mu.cpu())))))
        else:
            self.decompose_C_freq = 1

        # Use the SinglePopulationAlgorithmMixin to enable additional status reports regarding the population.
        SinglePopulationAlgorithmMixin.__init__(self)

        self.mu = mu_size  # Set the mu size for CMA-ES

    def update_mu_size(
        self,
        c_m: Real = 1.0,
        c_sigma: Optional[Real] = None,
        c_sigma_ratio: Real = 1.0,
        damp_sigma: Optional[Real] = None,
        damp_sigma_ratio: Real = 1.0,
        c_c: Optional[Real] = None,
        c_c_ratio: Real = 1.0,
        c_1: Optional[Real] = None,
        c_1_ratio: Real = 1.0,
        c_mu: Optional[Real] = None,
        c_mu_ratio: Real = 1.0,
        limit_C_decomposition: bool = True,
        mu_size: Optional[int] = None
    ):
        """
        Update the mu size and recompute the search parameters according to the CMA-ES rule of thumb if necessary.
        """
        self.mu = mu_size
        d = self._problem.solution_length

        # === Initialize raw weights ===
        # Conditioned on popsize

        # w_i = log((lambda + 1) / 2) - log(i) for i = 1 ... lambda
        # positive valued weights are the first mu
        positive_weights = self.raw_weights[: self.mu]
        negative_weights = self.raw_weights[self.mu :]

        # Variance effective selection mass of positive weights
        # Not affected by future updates to raw_weights
        self.mu_eff = torch.sum(positive_weights).pow(2.0) / torch.sum(positive_weights.pow(2.0))
        
        # Learning rate for step-size adaption
        if c_sigma is None:
            c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3)
        self.c_sigma = c_sigma_ratio * c_sigma

        # Damping factor for step-size adapation
        if damp_sigma is None:
            damp_sigma = 1 + 2 * max(0, torch.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.damp_sigma = damp_sigma_ratio * damp_sigma

        # Learning rate for evolution path for rank-1 update
        if c_c is None:
            # Branches on separability
            if self.separable:
                c_c = (1 + (1 / d) + (self.mu_eff / d)) / (d**0.5 + (1 / d) + 2 * (self.mu_eff / d))
            else:
                c_c = (4 + self.mu_eff / d) / (d + (4 + 2 * self.mu_eff / d))
        self.c_c = c_c_ratio * c_c

        # Learning rate for rank-1 update to covariance matrix
        if c_1 is None:
            # Branches on separability
            if self.separable:
                c_1 = 1.0 / (d + 2.0 * np.sqrt(d) + self.mu_eff / d)
            else:
                c_1 = min(1, self.popsize / 6) * 2 / ((d + 1.3) ** 2.0 + self.mu_eff)
        self.c_1 = c_1_ratio * c_1

        # Learning rate for rank-mu update to covariance matrix
        if c_mu is None:
            # Branches on separability
            if self.separable:
                c_mu = (0.25 + self.mu_eff + (1.0 / self.mu_eff) - 2) / (d + 4 * np.sqrt(d) + (self.mu_eff / 2.0))
            else:
                c_mu = min(
                    1 - self.c_1, 2 * ((0.25 + self.mu_eff - 2 + (1 / self.mu_eff)) / ((d + 2) ** 2.0 + self.mu_eff))
                )
        self.c_mu = c_mu_ratio * c_mu

        # The 'variance aware' coefficient used for the additive component of the evolution path for sigma
        self.variance_discount_sigma = torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        # The 'variance aware' coefficient used for the additive component of the evolution path for rank-1 updates
        self.variance_discount_c = torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)

        # === Finalize weights ===
        # Conditioned on search parameters and raw weights

        # Positive weights always sum to 1
        positive_weights = positive_weights / torch.sum(positive_weights)

        if self.active:
            # Active CMA-ES: negative weights sum to alpha

            # Get the variance effective selection mass of negative weights
            mu_eff_neg = torch.sum(negative_weights).pow(2.0) / torch.sum(negative_weights.pow(2.0))

            # Alpha is the minimum of the following 3 terms
            alpha_mu = 1 + self.c_1 / self.c_mu
            alpha_mu_eff = 1 + 2 * mu_eff_neg / (self.mu_eff + 2)
            alpha_pos_def = (1 - self.c_mu - self.c_1) / (d * self.c_mu)
            alpha = min([alpha_mu, alpha_mu_eff, alpha_pos_def])

            # Rescale negative weights
            negative_weights = alpha * negative_weights / torch.sum(torch.abs(negative_weights))
        else:
            # Negative weights are simply zero
            negative_weights = torch.zeros_like(negative_weights)

        # Concatenate final weights
        self.weights = torch.cat([positive_weights, negative_weights], dim=-1)

        # How often to decompose C
        if self.limit_C_decomposition:
            self.decompose_C_freq = max(1, int(np.floor(_safe_divide(1, 10 * d * (self.c_1.cpu() + self.c_mu.cpu())))))
        else:
            self.decompose_C_freq = 1

    def sample_distribution(self, num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the population. All 3 representations of solutions are returned for easy calculations of updates.
        Note that the computation time of this operation of O(d^2 num_samples) unless separable, in which case O(d num_samples)
        Args:
            num_samples (Optional[int]): The number of samples to draw. If None, then the population size is used
        Returns:
            zs (torch.Tensor): A tensor of shape [num_samples, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [num_samples, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            xs (torch.Tensor): A tensor of shape [num_samples, d] of samples from the search space e.g. x_i ~ N(m, sigma^2 C)
        """
        if num_samples is None:
            num_samples = self.popsize

        # # Generate z values (num_samples, d)
        # zs = self._problem.make_gaussian(num_solutions=num_samples)
        # zs = generate_qmc_normal(num_samples=num_samples, dim=self._problem.solution_length).to(self._problem.device)

        # # Generate student-t instead of standard normal
        # zs = generate_student_t_distribution(num_samples=num_samples, dim=self._problem.solution_length, v=5.0).to(self._problem.device)

        # # Generate low-discrepancy normal samples
        # zs = generate_low_discrepancy_normal(num_samples=num_samples, dim=self._problem.solution_length)

        # Generate sigma points
        zs = generate_sigma_normal(num_samples=num_samples, dim=self._problem.solution_length)

        # Construct ys = A zs
        if self.separable:
            # In the separable case A is diagonal so is represented as a single vector
            ys = self.A.unsqueeze(0) * zs
        else:
            ys = (self.A @ zs.T).T
        # Construct xs = m + sigma ys
        xs = self.m.unsqueeze(0) + self.sigma * ys

        return zs, ys, xs