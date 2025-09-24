"""
Lorenz System Generation
"""

import numpy as np
from scipy.integrate import solve_ivp


def generate_lorenz(sigma=10.0, beta=8/3, rho=28.0, dt=0.01, T=50.0, init=(1.0, 1.0, 1.0)):
    """Generate Lorenz system x-component only"""
    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(lorenz, (0, T), init, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    traj = sol.y.T  # shape (N, 3)
    return traj[:, 0], t_eval  # return only x-dimension


def generate_lorenz_full(sigma=10.0, beta=8/3, rho=28.0, dt=0.01, T=20.0, init=(1.0, 1.0, 1.0)):
    """Generate full Lorenz system (x, y, z components)"""
    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(lorenz, (0, T), init, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    return sol.y.T, t_eval  # shape (N, 3), t_eval
