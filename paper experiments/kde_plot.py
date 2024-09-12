import json
import numpy as np
from methods.kde import ProbaDensityEstimator
from utils.data import euler_maruyama, plot_map_1d
from utils.data_controlled import ornstein_uhlenbeck_controlled, piecewise_constant_control

# Load the experiment settings
with open('selection_kde_1d.json', 'r') as file:
    settings = json.load(file)

# Extract the settings
n = settings['simulation_parameters']['n']
theta = settings['simulation_parameters']['theta']
sigma = settings['simulation_parameters']['sigma']
mu_0 = settings['simulation_parameters']['mean_0']
sigma_0 = settings['simulation_parameters']['std_0']
T = settings['simulation_parameters']['T']
n_steps = settings['simulation_parameters']['n_steps']
dt = settings['simulation_parameters']['dt']
T_tr = np.array(settings['simulation_parameters']['T_tr'])
n_paths = settings['simulation_parameters']['n_paths']
K = settings['simulation_parameters']['K']
U0s = np.array(settings['controls']['U0s'])
U1s = np.array(settings['controls']['U1s'])
S1s = np.array(settings['controls']['S1s'])
hyperparams = settings['best_params_per_k']


for k in range(K):
    def u(t):
        return piecewise_constant_control(t, float(U0s[k]), float(U1s[k]), float(S1s[k]))

    # Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
    b, sigma_func = ornstein_uhlenbeck_controlled(u=u, theta=theta, sigma=sigma)

    # Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
    X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths, T, n, mu_0, sigma_0)

    # Initialize the probability density estimator.
    estimator = ProbaDensityEstimator()

    # Choose the hyperparameters.
    gamma_t = hyperparams[str(k)]['best_params']['gamma_t']
    L_t = hyperparams[str(k)]['best_params']['L_t']
    mu_x = hyperparams[str(k)]['best_params']['mu_x']
    c_kernel = hyperparams[str(k)]['best_params']['c_kernel']

    estimator.gamma_t = gamma_t
    estimator.mu_x = mu_x
    estimator.L_t = L_t
    estimator.c_kernel = c_kernel
    estimator.T = T

    # Fit the probability density estimator to the sample paths.
    estimator.fit(T_tr=T_tr, X_tr=X_tr)

    # Predict the values of the probability density on a test set.
    t_interpolation = dt / 2  # Time offset between test and train times
    T_te = np.array([i * dt + t_interpolation for i in range(n_steps)]).reshape(-1, 1)

    # Generate n_x points x in [-R, R]^n.
    Q = 200
    X_te = np.random.uniform(-2, 2, size=(Q, n)).reshape((-1, 1, n))

    # Predictions.
    p_pred = estimator.predict(T_te=T_te, X_te=X_te)

    # Plot the predictions for the first 3 controls
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