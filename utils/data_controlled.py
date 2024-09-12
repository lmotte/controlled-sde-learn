import numpy as np


def piecewise_constant_control(t, v0, v1, t1):
    """
    Returns a piecewise constant control, where the control value switches between
    two constants, v0 and v1, at a specified time t1.

    Parameters:
        t (float): The current time.
        v0 (float): The control value before the switch time t1.
        v1 (float): The control value after the switch time t1.
        t1 (float): The time at which the control value switches from v0 to v1.

    Returns:
        float: The control value based on the current time t. It returns v0 if t < t1,
        and v1 if t >= t1.
    """
    return np.where(t < t1, v0, v1)


def smooth_piecewise_control(t, v0, v1, t1, transition_width=1.0):
    """
    Returns a smoothly transitioning control value between two constants, v0 and v1, using
    a sigmoid function for the transition.

    Parameters:
        t (float): The current time.
        v0 (float): The control value before the switch time t1.
        v1 (float): The control value after the switch time t1.
        t1 (float): The time at which the control value switches from v0 to v1.
        transition_width (float): The width of the transition. Higher values make the transition
                                  smoother. Default is 1.0.

    Returns:
        float: The control value smoothly transitioning between v0 and v1 based on the current time t.
    """

    # Computes a numerically stable sigmoid function to ensure the transition is smooth.
    def stable_sigmoid(x):
        z = np.exp(-np.abs(x))  # Avoid overflow
        sig = np.where(x >= 0, 1 / (1 + z), z / (1 + z))  # Compute sigmoid based on input
        return sig

    # Compute the transition using the sigmoid function
    transition_value = transition_width * (t - t1)
    sigmoid = stable_sigmoid(transition_value)

    # Linearly interpolate between v0 and v1 using the sigmoid value
    return v0 * (1 - sigmoid) + v1 * sigmoid


def sinusoidal_control(t, v):
    """
        Returns a sinusoidal control value based on time.

        Parameters:
            t (float): The current time.
            v (float): A scaling factor for the amplitude of the sinusoidal function.

        Returns:
            float: The control value at time t, computed as 10 * v * sin(t * π / 10).
    """
    return 10 * v * np.sin(t / 10 * np.pi)


def ornstein_uhlenbeck_controlled(u, theta, sigma):
    """
    Returns the drift and diffusion coefficients for a controlled Ornstein–Uhlenbeck process.

    Parameters:
        u (callable): Control function that defines the target value for mean reversion at each time t.
        theta (float): The rate of mean reversion. Larger values mean faster reversion to the control target.
        sigma (float): The volatility parameter (i.e., the diffusion coefficient).

    Returns:
        tuple: A tuple (b, sigma_func) where:
            - b(t, x): The drift coefficient representing mean reversion towards the control target.
            - sigma_func(t, x): The diffusion coefficient, which remains constant for this process.
    """

    # Drift function: mean-reverts to the control target u(t).
    def b(t, x):
        return theta * (u(t) - x)

    # Constant diffusion function (sigma is the volatility parameter).
    def sigma_func(t, x):
        return sigma

    return b, sigma_func


def dubins_controlled(u, v, sigma):
    """
    Returns the drift and diffusion coefficients for a controlled Dubins process,
    which describes motion in 2D space with a controlled direction of movement.

    Parameters:
        u (callable): Control function that defines the steering angle over time.
        v (float): Constant speed of the process.
        sigma (float): Diffusion coefficient that defines the randomness in the process.

    Returns:
        tuple: A tuple (b, sigma_func) where:
            - b(t, x): The drift coefficient (velocity) for the process in terms of x and y components.
            - sigma_func(t, x): The diffusion coefficient (constant for this process).
    """

    # Drift function: controls movement based on the velocity and control function u(t).
    def b(t, x):
        return v * np.array([np.cos(1 / 10 * u(t)), np.sin(1 / 10 * u(t))])

    # Constant diffusion function (sigma represents the diffusion coefficient).
    def sigma_func(t, x):
        return sigma

    return b, sigma_func
