import numpy as np


# Brownian Motion
# Mode: Brownian motion, step size is proportional to sqrt(dt), random with Gaussian distribution.
def brownian_motion(T, dt, H, W, sigma=1, base = None):
    """
    Brownian Motion
    Mode: Brownian motion, step size is proportional to sqrt(dt), random with Gaussian distribution.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    step_size = np.random.uniform(1, 100)  # Random step size
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    trajectory[0] = base
    for t in range(1, T):
        dx = np.random.normal(0, np.sqrt(dt) * sigma) * step_size
        dy = np.random.normal(0, np.sqrt(dt) * sigma) * step_size
        trajectory[t] = trajectory[t-1] + [dx, dy]
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None
    
    return trajectory

# Constant Velocity Motion
# Mode: Uniform motion with constant velocity (no acceleration).
def constant_velocity_motion(T, dt, H, W, vx=1, vy=1, base = None):
    """
    Constant Velocity Motion
    Mode: Uniform motion with constant velocity (no acceleration).
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    trajectory[0] = base
    for t in range(1, T):
        trajectory[t] = trajectory[t-1] + [vx * dt, vy * dt]
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None
    
    return trajectory

# Linear Acceleration Motion
# Mode: Linear motion with constant acceleration.
def linear_acceleration_motion(T, dt, H, W, acceleration=0.1, base = None):
    """
    Linear Acceleration Motion
    Mode: Linear motion with constant acceleration.
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    trajectory[0] = base
    velocity_x = np.random.uniform(-1, 1) * 10  # Initial velocity in x direction
    velocity_y = np.random.uniform(-1, 1) * 10  # Initial velocity in y direction

    for t in range(1, T):
        velocity_x += acceleration * dt  # Update velocity with acceleration
        velocity_y += acceleration * dt  # Update velocity with acceleration
        trajectory[t] = trajectory[t-1] + [velocity_x, velocity_y]

        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None

    return trajectory

# Circular Motion
# Mode: Motion along a circular path, with constant angular velocity.
def circular_motion(T, dt, H, W, radius=1, angular_velocity=1, base = None):
    """
    Circular Motion
    Mode: Motion along a circular path, with constant angular velocity.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(T):
        theta = angular_velocity * t * dt  # Angle
        trajectory[t, 0] = base[0] + radius * np.cos(theta)  # x coordinate
        trajectory[t, 1] = base[1] + radius * np.sin(theta)  # y coordinate
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None
    
    return trajectory

# Simple Pendulum Motion
# Mode: Pendulum-like motion with gravitational influence.
def simple_pendulum_motion(T, dt, H, W, amplitude=1, gravity=-9.8, length=1, base = None):
    """
    Simple Pendulum Motion
    Mode: Pendulum-like motion with gravitational influence.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    # Initial conditions
    theta_0 = amplitude  # initial angle (radians)
    angular_velocity = 0  # initial angular velocity
    angular_acceleration = 0  # initial angular acceleration
    
    for t in range(T):
        angular_acceleration = - (gravity / length) * np.sin(theta_0)  # Pendulum equation
        angular_velocity += angular_acceleration * dt
        theta_0 += angular_velocity * dt
        
        trajectory[t, 0] = base[0] + length * np.sin(theta_0)  # x position
        trajectory[t, 1] = base[1] - length * np.cos(theta_0)  # y position
    
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None
    
    return trajectory

# Sine Wave Driven Motion
# Mode: Motion driven by a sinusoidal function in one direction (e.g., x direction).
def sine_wave_driven_motion(T, dt, H, W, amplitude=1, frequency=1, base = None):
    """
    Sine Wave Driven Motion
    Mode: Motion driven by a sinusoidal function in one direction (e.g., x direction).
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/5*2), int(H/5*3))]
    vy = np.random.uniform(-1, 1) * 2  # Initial velocity in y direction
    
    for t in range(T):
        trajectory[t, 0] = base[0] + amplitude * np.sin(2 * np.pi * frequency * t * dt)
        if t == 0:
            trajectory[t, 1] = base[1]
        else:
            trajectory[t, 1] = trajectory[t-1, 1] + vy # Assume motion occurs only along x-axis

        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None

    return trajectory

# Spiral Motion
# Mode: Motion along a spiral path (expanding radius with constant angular velocity).
# NOTE: we abandon this mode because it violate law of conservation of angular momentum
def spiral_motion(T, dt, H, W, radius=1, angular_velocity=1, expansion_rate=0.01, base = None):
    """
    Spiral Motion
    Mode: Motion along a spiral path (expanding radius with constant angular velocity).
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(T):
        theta = angular_velocity * t * dt  # Angle
        radius_t = radius + expansion_rate * t  # Expanding radius
        trajectory[t, 0] = base[0] + radius_t * np.cos(theta)  # x position
        trajectory[t, 1] = base[1] + radius_t * np.sin(theta)  # y position

        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None

    return trajectory

def spiral_motion_conserved(T, dt, H, W, radius=1, expansion_rate=0.01, base=None, v=100):
    """
    Spiral Motion (with conserved angular momentum)
    Mode: Motion along a spiral path, with constant angular velocity.
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    if base is None:
        base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    # Initialize radius and angular velocity
    r_0 = radius
    trajectory[0] = base
    omega_t = 0
    
    # The spiral motion is now governed by the condition that angular momentum is conserved.
    for t in range(1, T):
        # Compute the new radius at time t (growing exponentially)
        r_t = r_0 * np.exp(expansion_rate * t * dt)
        
        # Angular velocity is adjusted based on the current radius to keep constant speed
        d_omega = v / r_t  # Angular velocity based on constant linear speed
        omega_t += d_omega * dt
        
        # Update the trajectory
        trajectory[t, 0] = base[0] + r_t * np.cos(omega_t)
        trajectory[t, 1] = base[1] + r_t * np.sin(omega_t)
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return None

    return trajectory

def noised_motion(trajectory, noise_sigma=0.01):
    """add noise to the trajectory"""
    
    noise = np.random.normal(0, noise_sigma, trajectory.shape)
    noisy_trajectory = trajectory + noise
    
    return noisy_trajectory

def sin_r(T, bias, w, a = 0.1):
    t = np.arange(T)
    r = w * np.sin(a*t) + bias
    return noised_motion(r)
