import numpy as np
import cv2
from inanimate_types import *


class inanimate_generater:
    def __init__(self, T, dt, H, W, DATADIR):
        self.T = T
        self.dt = dt
        self.H = H
        self.W = W
        self.datadir = DATADIR
        
    def sample(self, motion_type):
        if motion_type == "brownian_motion":
            trajectory = brownian_motion(self.T, self.dt, self.H, self.W, sigma=1)
        elif motion_type == "constant_velocity":
            vx_ = np.random.uniform(-1, 1) * 100
            vy_ = np.random.uniform(-1, 1) * 100
            trajectory = constant_velocity_motion(self.T, self.dt, self.H, self.W, vx=vx_, vy=vy_)
        elif motion_type == "linear_acceleration":
            acc = np.random.uniform(-1, 1) * 1.5
            trajectory = linear_acceleration_motion(self.T, self.dt, self.H, self.W, acceleration=acc)
        elif motion_type == "circular":
            radius_ = np.random.uniform(50, 300)
            angular_vel = np.random.uniform(-1, 1) * 5
            trajectory = circular_motion(self.T, self.dt, self.H, self.W, radius=radius_, angular_velocity=angular_vel)
        elif motion_type == "simple_pendulum":
            len_ = np.random.uniform(100, 500)
            amplitude_ = np.random.uniform(-5, 5)
            trajectory = simple_pendulum_motion(self.T, self.dt, self.H, self.W, length=len_, amplitude=amplitude_, gravity=-980)
        elif motion_type == "sine_wave_driven":
            amplitude_ = np.random.uniform(100, 500)
            freq_ = np.random.uniform(0.1, 1)
            trajectory = sine_wave_driven_motion(self.T, self.dt, self.H, self.W, amplitude=amplitude_, frequency=freq_)
        elif motion_type == "spiral":
            radius_ = np.random.uniform(1, 5)
            angular_vel_ = np.random.uniform(-1, 1) * 100
            expansion_rate_ = np.random.uniform(0.2, 2)
            trajectory = spiral_motion(self.T, self.dt, self.H, self.W, radius=1, angular_velocity=angular_vel_, expansion_rate=expansion_rate_)
        else:
            raise ValueError("Invalid motion type")
        return trajectory
    
    def generate(self, motion_type, noise_sigma: float = 1, max_num: int = 100):
        flag = False
        for _ in range(max_num):
            trajectory = self.sample(motion_type)
            if trajectory is not None:
                flag = True
                break
        if not flag:
            print(f"Failed to generate a valid trajectory for {motion_type}")
            return False
        noisy_trajectory = noised_motion(trajectory, noise_sigma)
        r = sin_r(self.T, bias=25, w=10)
        noisy_trajectory = np.concatenate([noisy_trajectory, r[:, np.newaxis]], axis=1)
        # self.vis_without_trail_2(noisy_trajectory, f"{self.datadir}/{motion_type}.mp4")
        return noisy_trajectory
        
    def vis_with_trail(self, trajectory, video_filename, max_trail_time=10, frame_rate=60):
        """
        Create a video of the trajectory with a trailing effect and save it to the file.
        
        Parameters:
        - trajectory: (T, 3) trajectory data, T is the number of frames, and 3 is the (x, y, r) coordinates per frame.
        - video_filename: Filename to save the video.
        - max_trail_time: Maximum time for the trail to persist, controlling the trail effect.
        - frame_rate: Video frame rate, controlling the frame rate of the generated video.
        """
        
        # Initialize video writer with lower quality settings (e.g., use 'MJPG' codec)
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H264 codec for MP4 format
        out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (self.W, self.H))
        
        # Create a white background for the frames
        background = np.ones((self.H, self.W, 3), dtype=np.uint8)
        
        for t in range(len(trajectory)):
            # Copy the white background to prepare for drawing
            frame = background.copy()
            
            # Draw past points with decreasing opacity (trail effect)
            for i in range(t + 1):
                alpha = max(0, 1 - (t - i) / max_trail_time)  # Calculate alpha for transparency
                x, y, r = trajectory[i]  # x, y coordinates and the radius r
                x, y, r = int(x), int(y), int(r)  # Ensure x, y, r are integers
                
                if 0 < x < self.W-r and 0 < y < self.H-r:
                    # Draw a circle around the point with radius 'r' and apply transparency to the past points
                    # Create a mask for the transparent circle
                    overlay = frame.copy()
                    # Draw a transparent circle (alpha blending)
                    cv2.circle(overlay, (x, y), r, (0, 0, 0), -1)  # Fill the circle with black
                    # Blend the overlay with the frame using alpha transparency
                    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            
            # Draw the current point in black (completely opaque) using radius 'r'
            x, y, r = trajectory[t]
            x, y, r = int(x), int(y), int(r)
            
            if 0 <= x < self.W and 0 <= y < self.H:
                cv2.circle(frame, (x, y), r, (0, 0, 0), -1)  # Draw the current point with radius r

            # Write the current frame to the video file
            out.write(frame)
        
        # Release the video writer object
        out.release()
        print(f"Video saved as {video_filename}")
        
    def vis_without_trail_2(self, trajectory, video_filename, frame_rate=60):
        """
        Create a video of the trajectory and save it to the file.
        
        Parameters:
        - trajectory: (T, 6) trajectory data, T is the number of frames, and 6 is the (x1, y1, r1, x2, y2, r2) coordinates per frame.
        - video_filename: Filename to save the video.
        - frame_rate: Video frame rate, controlling the frame rate of the generated video.
        """
        
        # Initialize video writer with lower quality settings (e.g., use 'MJPG' codec)
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H264 codec for MP4 format
        out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (self.W, self.H))
        
        # Create a white background for the frames
        background = np.ones((self.H, self.W, 3), dtype=np.uint8) * 0
        
        # Define colors for the two points
        color_1 = (0, 0, 255)  # Blue color for the first point (BGR format)
        color_2 = (255, 0, 0)  # Red color for the second point (BGR format)
        
        for t in range(len(trajectory)):
            # Copy the white background to prepare for drawing
            frame = background.copy()
            
            # Extract coordinates and radius for both points at time t
            x1, y1, r1, x2, y2, r2 = trajectory[t]
            
            # Convert to integer coordinates and radius
            x1, y1, r1 = int(x1), int(y1), int(r1)
            x2, y2, r2 = int(x2), int(y2), int(r2)
            
            # Draw the first point (in blue)
            if 0 <= x1 < self.W and 0 <= y1 < self.H:
                cv2.circle(frame, (x1, y1), r1, color_1, -1)  # Draw the first point with radius r1
            
            # Draw the second point (in red)
            if 0 <= x2 < self.W and 0 <= y2 < self.H:
                cv2.circle(frame, (x2, y2), r2, color_2, -1)  # Draw the second point with radius r2

            # Write the current frame to the video file
            out.write(frame)
        
        # Release the video writer object
        out.release()
        print(f"Video saved as {video_filename}")
