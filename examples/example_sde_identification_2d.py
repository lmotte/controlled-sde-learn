import time
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import cartesian_products_of_rows
from methods.kde import ProbaDensityEstimator
from methods.fp_estimator_controlled import FPEstimator
from utils.data import (
    euler_maruyama,
    plot_paths_2d,
    line_plot,
    plot_map_2d
)
from utils.data_controlled import dubins_controlled, sinusoidal_control


# Define a function to create a sinusoidal control function.
def create_control_function(w):
    return lambda t: sinusoidal_control(t, w)


# Define function to set up and fit KDE estimators.
def setup_and_fit_estimator(Ts, Xs, gamma_t, L_t, mu_x, c_kernel, T_h):
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


# Simulation parameters for the controlled Dubins process
n = 2  # Dimension of the state variable
v = 2.0  # Constant velocity
sigma = 0.3  # Volatility (diffusion coefficient)
mean_0, std_0 = np.array([0] * n), 0.5  # Initial mean and standard deviation
T = 10  # Time horizon for the simulation
n_steps = 100  # Number of discretized time steps
dt = T / n_steps  # Time step size
T_tr = (np.arange(0, n_steps) * dt).reshape(-1, 1)  # Temporal discretization
n_paths_tr = 3000  # Number of train paths to draw

# Generate sample paths from the SDE for K different control functions.
K = 20  # Number of control functions to generate
Us = np.random.uniform(-1.2, 1.2, K)  # Random control function parameters

# Prepare for KDE fitting.
kde_list = []  # List to store KDE estimators
x_1_min, x_2_min, x_1_max, x_2_max = float('inf'), float('inf'), float('-inf'), float('-inf')  # Track x-axis limits
U_tr = []  # Store training control functions
all_controls_tr = np.zeros((K, n_steps, n))  # Preallocate space for controls

# Preallocate space for training sample paths used for Fokker-Planck estimation.
n_t_fp = 100
n_x_fp = 5
X_tr_fp_all = np.zeros((n_paths_tr, n_t_fp, n, K))

for k in range(K):
    # Create control function.
    u = create_control_function(float(Us[k]))
    U_tr.append(u)

    # Drift and diffusion coefficients for the Dubins process with control u.
    b, sigma_func = dubins_controlled(u=u, v=v, sigma=sigma)

    # Generate a training data set of sample paths from the SDE associated to the provided coefficients (b, sigma).
    X_tr = euler_maruyama(b, sigma_func, n_steps, n_paths_tr, T, n, mean_0, std_0)
    X_tr_fp = euler_maruyama(b, sigma_func, n_t_fp, n_paths_tr, T, n, mean_0, 5 * std_0)
    X_tr_fp_all[:, :, :, k] = X_tr_fp

    # Update x-axis limits
    x_1_min, x_1_max = min(X_tr[:, :, 0].min(), x_1_min) - 1, max(X_tr[:, :, 0].max() + 1, x_1_max)
    x_2_min, x_2_max = min(X_tr[:, :, 1].min() - 1, x_2_min), max(X_tr[:, :, 1].max() + 1, x_2_max)

    # Fit the probability density estimator to the sample paths.
    kde_list.append(setup_and_fit_estimator(T_tr, X_tr, 0.01, 1e-06, 4.6415888336127775, 1e-05, T))

    # Generate a uniform grid on [0,T] x [-beta/2, beta/2]^n.
    n_t_grid = 50
    n_x_grid = 1000
    n_x_grid = int(n_x_grid ** (1 / 2))
    dt_grid = T / n_t_grid
    t_interpolation = dt / 2  # Time offset between test and train times
    T_grid = np.array([i * dt_grid + t_interpolation for i in range(n_t_grid)]).reshape(
        -1, 1
    )

    X_1_grid = np.linspace(x_1_min, x_1_max, n_x_grid)
    X_2_grid = np.linspace(x_2_min, x_2_max, n_x_grid)
    X_grid = cartesian_products_of_rows(X_1_grid.reshape(-1, 1), X_2_grid.reshape(-1, 1))

    # Predict the probability density on the uniform grid.
    t0 = time.time()
    p_pred_grid = kde_list[-1](T_train=T_grid, X=X_grid, partial=False)
    print(f"KDE prediction time: {time.time() - t0}")

    # Plot the predicted probability density values.
    plot_map_2d(
        T_grid,
        X_1_grid,
        X_2_grid,
        f"p_pred_{k}",
        r"",
        xlabel="$x_1$",
        ylabel="$x_2$",
        alt_label=r"$t$",
        map_v=p_pred_grid,
        save_path="../plots/test_sde_identification_2d/",
        x_lim=(x_1_min, x_1_max),
        y_lim=(x_2_min, x_2_max)
    )

# Plot the data set of all controls.
fig, ax = plt.subplots(figsize=(8, 5))
for k in range(K):
    ax.plot(T_tr, all_controls_tr[k], color="black")
ax.set_xlabel("t", fontsize=18)
ax.set_ylabel("u(t)", fontsize=18, labelpad=20)
[spine.set_linewidth(1.5) for spine in ax.spines.values()]
ax.tick_params(axis="both", which="major", labelsize=18, width=1.5)
ax.tick_params(axis="both", which="minor", labelsize=18, width=1.5)
ax.grid(True, linestyle="--", alpha=0.5)
save_path = f"../plots/test_sde_identification_2d/controls_tr_all.pdf"
x_ticks = [0, 2, 4, 6, 8, 10]
y_ticks = [-10, -5, 0, 5, 10]
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
fig.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close(fig)

# Initialize the Fokker-Planck matching estimator.
estimator = FPEstimator()

# Set hyperparameters for Fokker-Planck matching estimator
gamma_z = 0.005  # Temporal kernel width for Fokker-Planck estimator
c_kernel_z = 1e-05  # Regularization parameter
la = 1e-07  # Regularization parameter for FP estimator
estimator.gamma_z = gamma_z
estimator.la = la
estimator.T = T
estimator.d = 1
estimator.c_kernel = c_kernel_z

# Select training points for Fokker-Planck estimation
dt_fp = T / n_t_fp  # Time step
T_fp = (np.arange(0, n_t_fp) * dt_fp).reshape(-1, 1)  # Temporal discretization
X_fp = np.zeros((n_x_fp, n_t_fp, n, K))
for i in range(n_t_fp):
    sub_idx_x = np.random.choice(n_paths_tr, size=n_x_fp, replace=False)
    X_fp_i = X_tr_fp_all[sub_idx_x, i, :, :]
    X_fp[:, i, :, :] = X_fp_i

# Plot the predicted probability density values with selected subset of the training set for Fokker-Planck matching.
# Generate a uniform grid on [0,T] x [-beta/2, beta/2]^n.
n_t_grid = 50
n_x_grid = 1000
n_x_grid = int(n_x_grid ** (1 / 2))
dt_grid = T / n_t_grid
t_interpolation = dt / 2  # Time offset between test and train times
T_grid = np.array([i * dt_grid + t_interpolation for i in range(n_t_grid)]).reshape(
    -1, 1
)
X_1_grid = np.linspace(x_1_min, x_1_max, n_x_grid)
X_2_grid = np.linspace(x_2_min, x_2_max, n_x_grid)
X_grid = cartesian_products_of_rows(X_1_grid.reshape(-1, 1), X_2_grid.reshape(-1, 1))
t0 = time.time()
p_pred_grid = kde_list[0](T_train=T_grid, X=X_grid, partial=False)
print(f"KDE prediction time: {time.time() - t0}")
plot_map_2d(
    T_grid,
    X_1_grid,
    X_2_grid,
    f"FP_training_set",
    r"Fokker-Planck training set",
    xlabel="$x_1$",
    ylabel="$x_2$",
    alt_label=r"$t$",
    map_v=p_pred_grid,
    save_path="../plots/test_sde_identification_2d/",
    T_plot=T_fp,
    X_plot=X_fp,
)

# Generate K_te different control functions drawn randomly.
K_te = 5
Us = [-1, -1 / 2, 0, 1 / 2, 1]
Us = np.array(Us)
all_paths = np.zeros((n_paths_tr * len(Us), n_steps, n))

U_te = []
kde_list_te = []

for k in range(K_te):
    u = create_control_function(Us[k])
    U_te.append(u)

    n_t_plot = 100
    dt_plot = T / n_t_plot
    T_plot = np.array([i * dt_plot for i in range(n_t_plot)]).reshape(-1, 1)
    Vs = [u(t) for t in T_plot]
    x_ticks = [0, 2, 4, 6, 8, 10]
    y_ticks = [-10, -5, 0, 5, 10]
    line_plot(T_plot, Vs, save_path=f"../plots/test_sde_identification_2d/control_te_{k}.pdf", title=f"",
              xlabel="t", ylabel="u(t)", fontsize=24, linewidth=3, x_ticks=x_ticks, y_ticks=y_ticks)

# Fit the  Fokker-Planck matching estimator with the training samples.
estimator.fit(T_tr=T_fp, X_tr=X_fp, U_tr=U_tr, kde_list=kde_list)

# Generate a set of sample paths from the SDE associated to the estimated coefficients (b, sigma).
print("Start sampling")
n_paths = 100
n_steps = 100
t0 = time.time()
path_pos = []
path_true = []
x_1_min, x_2_min, x_1_max, x_2_max = float('inf'), float('inf'), float('-inf'), float('-inf')

T = 10

for k in range(K_te):
    # Define estimated function
    def b(t, x):
        b_v = estimator.predict(
            T_te=np.array(t).reshape(1, 1),
            X_te=np.array(x).reshape(1, n),
            V_te=np.array(U_te[k](t)).reshape(1, 1)
        )[0].reshape((2,))
        return b_v


    def sigma_func(t, x):
        s = estimator.predict(
            T_te=np.array(t).reshape(1, 1),
            X_te=np.array(x).reshape(1, n),
            V_te=np.array(U_te[k](t)).reshape(1, 1),
            thresholding=True,
        )[1][0, 0]
        return s


    # Drift and diffusion coefficients for the Ornstein–Uhlenbeck process.
    b_true, sigma_func_true = dubins_controlled(u=U_te[k], v=v, sigma=sigma)

    path_pos_k = euler_maruyama(
        b, sigma_func, n_steps, n_paths, T, n, mean_0, std_0, time=True
    )
    path_pos.append(path_pos_k)

    path_pos_k_true = euler_maruyama(
        b_true, sigma_func_true, n_steps, n_paths, T, n, mean_0, std_0, time=True
    )
    path_true.append(path_pos_k_true)

    x_1_min, x_1_max = min(path_pos_k_true[:, :, 0].min(), x_1_min) - 3, max(path_pos_k_true[:, :, 0].max() + 3,
                                                                             x_1_max)
    x_2_min, x_2_max = min(path_pos_k_true[:, :, 1].min() - 1, x_2_min) - 3, max(path_pos_k_true[:, :, 1].max() + 3,
                                                                                 x_2_max)

print(f"End sampling")
print(f"Sampling computation time: {time.time() - t0}")

# Plot the set.
x_1_min, x_2_min, x_1_max, x_2_max = -5, -15, 25, 15

for k in range(K_te):
    dt = T / n_steps
    T_samp = (np.arange(0, n_steps) * dt).reshape(-1, 1)
    save_path = f"../plots/test_sde_identification_2d/samples_estimated_2d_{k}.pdf"
    plot_paths_2d(
        path_pos[k],
        save_path=save_path,
        x_lim=(x_1_min, x_1_max),
        y_lim=(x_2_min, x_2_max))

    save_path = f"../plots/test_sde_identification_2d/samples_true_2d_{k}.pdf"
    plot_paths_2d(
        path_true[k],
        save_path=save_path,
        x_lim=(x_1_min, x_1_max),
        y_lim=(x_2_min, x_2_max))
