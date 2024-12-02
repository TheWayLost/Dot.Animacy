import numpy as np


# Brownian Motion
# Mode: Brownian motion, step size is proportional to sqrt(dt), random with Gaussian distribution.
def brownian_motion(T, dt, H, W, sigma=1, step_size=1):
    """
    Brownian Motion
    Mode: Brownian motion, step size is proportional to sqrt(dt), random with Gaussian distribution.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    trajectory[0] = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(1, T):
        dx = np.random.normal(0, np.sqrt(dt) * sigma) * step_size
        dy = np.random.normal(0, np.sqrt(dt) * sigma) * step_size
        trajectory[t] = trajectory[t-1] + [dx, dy]
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return brownian_motion(T, dt, H, W, sigma, step_size)
    
    return trajectory

# Constant Velocity Motion
# Mode: Uniform motion with constant velocity (no acceleration).
def constant_velocity_motion(T, dt, H, W, vx=1, vy=1, step_size=1):
    """
    Constant Velocity Motion
    Mode: Uniform motion with constant velocity (no acceleration).
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    trajectory[0] = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(1, T):
        trajectory[t] = trajectory[t-1] + [vx * dt * step_size, vy * dt * step_size]
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return constant_velocity_motion(T, dt, H, W, vx, vy, step_size)
    
    return trajectory

# Linear Acceleration Motion
# Mode: Linear motion with constant acceleration.
def linear_acceleration_motion(T, dt, H, W, acceleration=0.1, step_size=1):
    """
    Linear Acceleration Motion
    Mode: Linear motion with constant acceleration.
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    trajectory[0] = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    velocity_x = 0
    velocity_y = 0

    for t in range(1, T):
        velocity_x += acceleration * dt  # Update velocity with acceleration
        velocity_y += acceleration * dt  # Update velocity with acceleration
        trajectory[t] = trajectory[t-1] + [velocity_x * step_size, velocity_y * step_size]

        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return linear_acceleration_motion(T, dt, H, W, acceleration, step_size)

    return trajectory

# Oscillatory Motion (Simple Harmonic Motion)
# Mode: Periodic oscillation, resembling simple harmonic motion.
def oscillatory_motion(T, dt, H, W, amplitude=1, frequency=1, step_size=1):
    """
    Oscillatory Motion
    Mode: Periodic oscillation, resembling simple harmonic motion.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(T):
        trajectory[t, 0] = base[0] + amplitude * np.sin(2 * np.pi * frequency * t * dt) * step_size
        trajectory[t, 1] = base[1]  # Assume oscillation only in x direction
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return oscillatory_motion(T, dt, H, W, amplitude, frequency, step_size)
    
    return trajectory

# Circular Motion
# Mode: Motion along a circular path, with constant angular velocity.
def circular_motion(T, dt, H, W, radius=1, angular_velocity=1, step_size=1):
    """
    Circular Motion
    Mode: Motion along a circular path, with constant angular velocity.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(T):
        theta = angular_velocity * t * dt  # Angle
        trajectory[t, 0] = base[0] + radius * np.cos(theta) * step_size  # x coordinate
        trajectory[t, 1] = base[1] + radius * np.sin(theta) * step_size  # y coordinate
        
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return circular_motion(T, dt, H, W, radius, angular_velocity, step_size)
    
    return trajectory

# Simple Pendulum Motion
# Mode: Pendulum-like motion with gravitational influence.
def simple_pendulum_motion(T, dt, H, W, amplitude=1, gravity=-9.8, length=1, step_size=1):
    """
    Simple Pendulum Motion
    Mode: Pendulum-like motion with gravitational influence.
    The trajectory is restricted within the bounds (H, W).
    """
    # Initialize position at a random point within bounds
    trajectory = np.zeros((T, 2))
    base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    # Initial conditions
    theta_0 = amplitude  # initial angle (radians)
    angular_velocity = 0  # initial angular velocity
    angular_acceleration = 0  # initial angular acceleration
    
    for t in range(T):
        angular_acceleration = - (gravity / length) * np.sin(theta_0)  # Pendulum equation
        angular_velocity += angular_acceleration * dt
        theta_0 += angular_velocity * dt
        
        trajectory[t, 0] = base[0] + length * np.sin(theta_0) * step_size  # x position
        trajectory[t, 1] = base[1] - length * np.cos(theta_0) * step_size  # y position
    
        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return simple_pendulum_motion(T, dt, H, W, amplitude, gravity, length, step_size)
    
    return trajectory

# Sine Wave Driven Motion
# Mode: Motion driven by a sinusoidal function in one direction (e.g., x direction).
def sine_wave_driven_motion(T, dt, H, W, amplitude=1, frequency=1, step_size=1):
    """
    Sine Wave Driven Motion
    Mode: Motion driven by a sinusoidal function in one direction (e.g., x direction).
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/5*2), int(H/5*3))]
    vy = np.random.uniform(-1, 1) * 1  # Initial velocity in y direction
    

    for t in range(T):
        trajectory[t, 0] = base[0] + amplitude * np.sin(2 * np.pi * frequency * t * dt) * step_size
        if t == 0:
            trajectory[t, 1] = base[1]
        trajectory[t, 1] = trajectory[t-1, 1] + vy * step_size # Assume motion occurs only along x-axis

        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return sine_wave_driven_motion(T, dt, H, W, amplitude, frequency, step_size)

    return trajectory

# Spiral Motion
# Mode: Motion along a spiral path (expanding radius with constant angular velocity).
def spiral_motion(T, dt, H, W, radius=1, angular_velocity=1, expansion_rate=0.01, step_size=1):
    """
    Spiral Motion
    Mode: Motion along a spiral path (expanding radius with constant angular velocity).
    The trajectory is restricted within the bounds (H, W).
    """
    trajectory = np.zeros((T, 2))
    base = [np.random.uniform(int(W/5*2), int(W/5*3)), np.random.uniform(int(H/4), int(H/4*3))]
    
    for t in range(T):
        theta = angular_velocity * t * dt  # Angle
        radius_t = radius + expansion_rate * t  # Expanding radius
        trajectory[t, 0] = base[0] + radius_t * np.cos(theta) * step_size  # x position
        trajectory[t, 1] = base[1] + radius_t * np.sin(theta) * step_size  # y position

        # Ensure trajectory stays within bounds
        if trajectory[t, 0] < 0 or trajectory[t, 0] > W or trajectory[t, 1] < 0 or trajectory[t, 1] > H:
            return spiral_motion(T, dt, H, W, radius, angular_velocity, expansion_rate, step_size)

    return trajectory

def noised_motion(trajectory, noise_sigma=0.01):
    """add noise to the trajectory"""
    
    noise = np.random.normal(0, noise_sigma, trajectory.shape)
    noisy_trajectory = trajectory + noise
    
    return noisy_trajectory
