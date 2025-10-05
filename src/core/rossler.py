"""
Rössler Attractor Generation
Generates the Rössler attractor using the standard equations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_rossler_full(T=20.0, dt=0.01, a=0.2, b=0.2, c=5.7, initial_conditions=None):
    """
    Generate the full Rössler attractor
    
    Args:
        T (float): Total time
        dt (float): Time step
        a, b, c (float): Rössler parameters
        initial_conditions (list): Initial conditions [x0, y0, z0]
    
    Returns:
        traj (np.array): Trajectory array (n_points, 3)
        t (np.array): Time array
    """
    if initial_conditions is None:
        initial_conditions = [1.0, 1.0, 1.0]
    
    # Time array
    t = np.arange(0, T, dt)
    n_points = len(t)
    
    # Initialize trajectory
    traj = np.zeros((n_points, 3))
    traj[0] = initial_conditions
    
    # Rössler equations
    def rossler_equations(x, y, z, a, b, c):
        dx_dt = -y - z
        dy_dt = x + a * y
        dz_dt = b + z * (x - c)
        return dx_dt, dy_dt, dz_dt
    
    # Numerical integration (4th order Runge-Kutta)
    for i in range(n_points - 1):
        x, y, z = traj[i]
        
        # k1
        k1x, k1y, k1z = rossler_equations(x, y, z, a, b, c)
        
        # k2
        k2x, k2y, k2z = rossler_equations(x + dt*k1x/2, y + dt*k1y/2, z + dt*k1z/2, a, b, c)
        
        # k3
        k3x, k3y, k3z = rossler_equations(x + dt*k2x/2, y + dt*k2y/2, z + dt*k2z/2, a, b, c)
        
        # k4
        k4x, k4y, k4z = rossler_equations(x + dt*k3x, y + dt*k3y, z + dt*k3z, a, b, c)
        
        # Update
        traj[i+1, 0] = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        traj[i+1, 1] = y + dt*(k1y + 2*k2y + 2*k3y + k4y)/6
        traj[i+1, 2] = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    
    return traj, t

def visualize_rossler_attractor(traj, t, save_path=None):
    """
    Visualize the Rössler attractor
    
    Args:
        traj (np.array): Trajectory array
        t (np.array): Time array
        save_path (str): Path to save the plot
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.8, linewidth=1, color='blue')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=100, label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Rössler Attractor (3D)')
    ax1.legend()
    
    # X-Y projection
    ax2 = fig.add_subplot(132)
    ax2.plot(traj[:, 0], traj[:, 1], alpha=0.8, linewidth=1, color='blue')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Rössler Attractor (X-Y)')
    ax2.grid(True, alpha=0.3)
    
    # Time series
    ax3 = fig.add_subplot(133)
    ax3.plot(t, traj[:, 0], alpha=0.8, linewidth=1, color='red', label='X')
    ax3.plot(t, traj[:, 1], alpha=0.8, linewidth=1, color='green', label='Y')
    ax3.plot(t, traj[:, 2], alpha=0.8, linewidth=1, color='blue', label='Z')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.set_title('Rössler Time Series')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rössler attractor visualization saved to: {save_path}")
    
    plt.show()
    return fig

def main():
    """Test Rössler attractor generation"""
    print("=== RÖSSLER ATTRACTOR GENERATION ===")
    
    # Generate Rössler attractor with longer duration for full state space development
    traj, t = generate_rossler_full(T=100.0, dt=0.01)  # Increased from 20.0 to 100.0
    
    print(f"Trajectory shape: {traj.shape}")
    print(f"Time range: {t[0]:.2f} to {t[-1]:.2f}")
    print(f"X range: [{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}]")
    print(f"Y range: [{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}]")
    print(f"Z range: [{traj[:, 2].min():.2f}, {traj[:, 2].max():.2f}]")
    
    # Visualize
    fig = visualize_rossler_attractor(traj, t, save_path='plots/rossler_attractor.png')
    
    return traj, t

if __name__ == "__main__":
    traj, t = main()
