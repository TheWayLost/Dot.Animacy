import numpy as np
import cv2
from inanimate_types import *


class inanimate_generater:
    def __init__(self, dt, H, W, DATADIR):
        self.dt = dt
        self.H = H
        self.W = W
        self.datadir = DATADIR
        
    def sample(self, motion_type, T, base=None):
        if motion_type == "brownian_motion":
            trajectory = brownian_motion(T, self.dt, self.H, self.W, sigma=1, base=base)
        elif motion_type == "constant_velocity":
            vx_ = np.random.uniform(-1, 1) * 80
            vy_ = np.random.uniform(-1, 1) * 80
            trajectory = constant_velocity_motion(T, self.dt, self.H, self.W, vx=vx_, vy=vy_, base=base)
        elif motion_type == "linear_acceleration":
            acc = np.random.uniform(-1, 1) * 2
            trajectory = linear_acceleration_motion(T, self.dt, self.H, self.W, acceleration=acc, base=base)
        elif motion_type == "circular":
            radius_ = np.random.uniform(50, 300)
            angular_vel = np.random.uniform(-1, 1) * 2
            trajectory = circular_motion(T, self.dt, self.H, self.W, radius=radius_, angular_velocity=angular_vel, base=base)
        elif motion_type == "simple_pendulum":
            len_ = np.random.uniform(100, 500)
            amplitude_ = np.random.uniform(-5, 5)
            trajectory = simple_pendulum_motion(T, self.dt, self.H, self.W, length=len_, amplitude=amplitude_, gravity=-980, base=base)
        elif motion_type == "sine_wave_driven":
            amplitude_ = np.random.uniform(100, 500)
            freq_ = np.random.uniform(0.1, 1)
            trajectory = sine_wave_driven_motion(T, self.dt, self.H, self.W, amplitude=amplitude_, frequency=freq_, base=base)
        elif motion_type == "spiral":
            radius_ = np.random.uniform(10, 100)
            expansion_rate_ = np.random.uniform(0.2, 3)
            lin_vel = np.random.uniform(-1, 1) * 500
            trajectory = spiral_motion_conserved(T, self.dt, self.H, self.W, radius=radius_, expansion_rate=expansion_rate_, base=base, v=lin_vel)
        else:
            raise ValueError("Invalid motion type")
        return trajectory
    
    def generate(self, motion_type, T: int = 360, max_num: int = 100, base = None):
        flag = False
        for _ in range(max_num):
            trajectory = self.sample(motion_type, T, base)
            if trajectory is not None:
                flag = True
                break
        if not flag:
            print(f"Failed to generate a valid trajectory for {motion_type}")
            return False
        # noisy_trajectory = trajectory
        noise_sigma = np.random.uniform(2, 5)
        noisy_trajectory = noised_motion(trajectory, noise_sigma)
        # r = 25
        r = sin_r(T, bias=25, w=10)
        noisy_trajectory = np.concatenate([noisy_trajectory, r[:, np.newaxis]], axis=1)
        # self.vis_without_trail_2(noisy_trajectory, f"{self.datadir}/{motion_type}.mp4")
        return noisy_trajectory