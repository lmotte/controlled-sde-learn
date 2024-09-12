import time
import numpy as np
from methods.kde import ProbaDensityEstimator
from methods.fp_estimator_controlled import FPEstimator
from utils.data import (
    euler_maruyama,
    plot_paths_1d,
)
from utils.data_controlled import ornstein_uhlenbeck_controlled, piecewise_constant_control


# Define a function to create a piecewise constant control function
def create_control_function(U0, U1, S1):
    """
        Creates a piecewise constant control function based on input parameters.

        Args:
            U0 (float): Initial control value.
            U1 (float): Final control value.
            S1 (float): Time at which the control switches from U0 to U1.

        Returns:
            function: A lambda function representing the control over time `t`.
        """
    return lambda t: piecewise_constant_control(t, float(U0), float(U1), float(S1))


# Define a function to set up and fit KDE estimators
def setup_and_fit_estimator(Ts, Xs, gamma_t, L_t, mu_x, c_kernel, T_h):
    """
        Sets up and fits a Kernel Density Estimator (KDE) to the provided training data.

        Args:
            Ts (np.ndarray): Temporal discretization (training time points).
            Xs (np.ndarray): Training data (sample paths).
            gamma_t (float): Temporal kernel width.
            L_t (float): Temporal regularization parameter.
            mu_x (float): Spatial kernel width.
            c_kernel (float): Regularization parameter for the kernel.
            T_h (float): Total time horizon for the estimator.

        Returns:
            function: A function to predict the probability density using the KDE.
        """
    est = ProbaDensityEstimator()
    est.gamma_t = gamma_t
    est.mu_x = mu_x
    est.L_t = L_t
    est.c_kernel = c_kernel
    est.T = T_h
    est.fit(T_tr=Ts, X_tr=Xs)

    def p(T_train, X, partial):
        return est.predict(T_train, X, partial=partial)

    return p


# Simulation parameters for the controlled Ornstein-Uhlenbeck process
n = 1  # Dimension of the state variable
theta = 0.5  # Rate of mean reversion
sigma = 0.5 * theta ** (1 / 2)  # Volatility (diffusion coefficient)
T = 10.0  # Time horizon for the simulation
mu_0, sigma_0 = np.array([0] * n), sigma / (2 * theta) ** (
        1 / 2
)  # # Initial conditions (mean and std at t=0)
n_steps = 100  # Number of discretized time steps
dt = T / n_steps  # Size of each time step
T_tr = (np.arange(1, n_steps + 1) * dt).reshape(-1, 1)  # Temporal discretization
n_paths = 1000  # Number of paths to draw

# Generate sample paths from the SDE for K different control functions
K = 10  # Number of different control functions
U0s = np.random.uniform(-2, 2, K)
U1s = np.random.uniform(-2, 2, K)
S1s = np.random.uniform(3, 7, K)
all_paths = np.zeros((n_paths * len(U1s), n_steps, n))

# Prepare for KDE fitting
kde_list = []
x_min, x_max = float('inf'), float('-inf')
U_tr = []

for k in range(K):
    # Create control function.
    u = create_control_function(float(U0s[k]), float(U1s[k]), float(S1s[k]))
    U_tr.append(u)

    # Define drift and diffusion coefficients for the controlled Ornstein-Uhlenbeck process.
    b, sigma_func = ornstein_uhlenbeck_controlled(u=u, theta=theta, sigma=sigma)

    # Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
    X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)

    # Update x-axis limits
    x_min = min(X_tr[:, :, 0].min() - 1, x_min)
    x_max = max(X_tr[:, :, 0].max() + 1, x_max)

    # Fit the probability density estimator to the sample paths.
    kde_list.append(setup_and_fit_estimator(T_tr, X_tr, 1, 1e-3, 10, 1e-5, T))

# Initialize the Fokker-Planck matching estimator.
estimator = FPEstimator()

# Set hyperparameters for Fokker-Planck matching estimator
gamma_z = 1e-1  # Spatial kernel width
c_kernel_z = 1e-5  # Spatial regularization parameter
la = 1e-5  # Regularization parameter
estimator.gamma_z = gamma_z
estimator.la = la
estimator.T = T
estimator.c_kernel = c_kernel_z

# Generate training points (t,x) uniformly in [0,T] x [-beta/2, beta/2]^n.
n_t_fp = 20
n_fp = 50
T_fp = np.random.uniform(0, T, size=(n_t_fp, 1))
X_fp = np.random.uniform(x_min, x_max, size=(n_fp, 1))

# Generate uniform grid of test points (t,x) in [0,T] x [-beta/2, beta/2]^n for beta > 0.
n_t_te = 50
n_x_te = 200
dt_te = T / n_t_te
t_interpolation = dt_te / 2  # Time offset for test points
T_te = np.array([i * dt_te + t_interpolation for i in range(n_t_te)]).reshape(-1, 1)
X_te = np.linspace(x_min, x_max, n_x_te).reshape(-1, 1)

# Generate K_te different control functions drawn randomly.
K_te = 10
U0s = np.random.uniform(-2, 2, K_te)
U1s = np.random.uniform(-2, 2, K_te)
S1s = np.random.uniform(3, 7, K_te)
all_paths = np.zeros((n_paths * len(U1s), n_steps, n))

x_min, x_max = float('inf'), float('-inf')
U_te = []
kde_list_te = []

for k in range(K):
    # Create control function for testing
    u = create_control_function(float(U0s[k]), float(U1s[k]), float(S1s[k]))
    U_te.append(u)

    # Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
    b, sigma_func = ornstein_uhlenbeck_controlled(u=u, theta=theta, sigma=sigma)

    # Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
    X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)
    x_min = min(X_tr[:, :, 0].min() - 1, x_min)
    x_max = max(X_tr[:, :, 0].max() + 1, x_max)

# Fit the  Fokker-Planck matching estimator with the training samples.
estimator.fit(T_tr=T_fp, X_tr=X_fp, U_tr=U_tr, kde_list=kde_list)

# Generate a set of sample paths from the SDE associated to the estimated coefficients (b, sigma).
print("Start sampling")
n_paths = 100
n_steps = 100
t0 = time.time()
path_pos = []
path_true = []

for k in range(K_te):
    # Define estimated function
    def b(t, x):
        b_pred = estimator.predict(
            T_te=np.array(t).reshape(1, 1), X_te=np.array(x).reshape(1, 1), V_te=np.array(U_te[k](t)).reshape(1, 1)
        )[0]
        return b_pred


    def sigma_func(t, x):
        s_pred = estimator.predict(
            T_te=np.array(t).reshape(1, 1),
            X_te=np.array(x).reshape(1, 1),
            V_te=np.array(U_te[k](t)).reshape(1, 1),
            thresholding=True,
        )[1]

        return s_pred


    # Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
    b_true, sigma_func_true = ornstein_uhlenbeck_controlled(u=U_te[k], theta=theta, sigma=sigma)

    # Simulate paths with estimated coefficients
    path_pos_k = euler_maruyama(
        b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0, time=True
    )
    path_pos.append(path_pos_k)

    # Simulate paths with true coefficients
    path_pos_k_true = euler_maruyama(
        b_true, sigma_func_true, n_steps, n_paths, T, n, mu_0, sigma_0, time=True
    )
    path_true.append(path_pos_k_true)

print(f"End sampling")
print(f"Sampling computation time: {time.time() - t0}")

# Plot the generated paths (estimated and true)
for k in range(K_te):
    dt = T / n_steps
    T_samp = (np.arange(0, n_steps) * dt).reshape(-1, 1)

    # Plot estimated sample paths
    save_path = f"../plots/test_sde_identification_1d/samples_estimated_{k}.pdf"
    plot_paths_1d(T_samp, path_pos[k], save_path=save_path)

    # Plot true sample paths
    save_path = f"../plots/test_sde_identification_1d/samples_true_{k}.pdf"
    plot_paths_1d(T_samp, path_true[k], save_path=save_path)
