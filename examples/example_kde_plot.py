import numpy as np
from methods.kde import ProbaDensityEstimator
from utils.data import (
    euler_maruyama,
    plot_map_1d,
)
from utils.data_controlled import ornstein_uhlenbeck_controlled, piecewise_constant_control

# Simulation parameters for the SDE.
n = 1  # Dimension of the state variable
theta = 1.  # Rate of mean reversion for the Ornstein-Uhlenbeck process
sigma = 0.2  # Volatility (diffusion coefficient)
T = 10.0  # Time horizon for the simulation
mu_0, sigma_0 = np.array([0] * n), sigma / (2 * theta) ** (
        1 / 2
)  # Initial conditions for the mean and variance
n_steps = 100  # Number of discretized time steps
dt = T / n_steps  # Time step size
T_tr = (np.arange(1, n_steps + 1) * dt).reshape(-1, 1)  # Temporal discretization
n_paths = 3000  # Number of paths to draw

# Generate sample paths from the SDE for K different control functions
K = 3  # Number of different random control functions
U0s = np.random.uniform(-2, 2, K)  # Random initial control values
U1s = np.random.uniform(-2, 2, K)  # Random final control values
S1s = np.random.uniform(3, 7, K)  # Random time control switches between U0 and U1
all_paths = np.zeros((n_paths * len(U1s), n_steps, n))  # Preallocate space for all paths

# Loop over different control functions.
for k in range(K):
    # Define the control function as a piecewise constant.
    def u(t):
        return piecewise_constant_control(t, float(U0s[k]), float(U1s[k]), float(S1s[k]))


    # Drift and diffusion coefficients for the Ornstein–Uhlenbeck process with control u_k.
    b, sigma_func = ornstein_uhlenbeck_controlled(u=u, theta=theta, sigma=sigma)

    # Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
    X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)

    # Initialize the probability density estimator.
    estimator = ProbaDensityEstimator()

    # Set hyperparameters for the probability density estimator.
    gamma_t = 1  # Temporal kernel width
    L_t = 1e-6  # Temporal regularization parameter
    mu_x = 7  # Spatial kernel width
    c_kernel = 1e-5  # Add small constant to kernel
    estimator.gamma_t = gamma_t
    estimator.mu_x = mu_x
    estimator.L_t = L_t
    estimator.c_kernel = c_kernel
    estimator.T = T

    # Fit the probability density estimator to the sample paths.
    estimator.fit(T_tr=T_tr, X_tr=X_tr)

    # Define test set times with a slight interpolation offset.
    t_interpolation = dt / 2  # Offset for test time points
    T_te = np.array([i * dt + t_interpolation for i in range(n_steps)]).reshape(-1, 1)

    # Generate n_x test points x in [-R, R]^n.
    Q = 200
    X_te = np.random.uniform(-2, 2, size=(Q, n)).reshape((-1, 1, n))

    # Predict the probability density on the test set.
    p_pred = estimator.predict(T_te=T_te, X_te=X_te)

    # Save the prediction plots for the first 3 control functions.
    save_path = "../plots/test_kde_plot/"

    if k < 3:
        plot_map_1d(
            T_te,
            X_te,
            f"p_pred_{k}",
            r"$\hat{p}$" + f" : $u_0$={U0s[k]:.1f} | $u_1$={U1s[k]:.1f} | $t_1$={S1s[k]:.1f}",
            xlabel="x",
            ylabel="",
            alt_label=r"$t$",
            map1=p_pred,
            save_path=save_path,
        )
