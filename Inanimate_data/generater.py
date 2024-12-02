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
            trajectory = brownian_motion(self.T, self.dt, self.H, self.W, sigma=1, step_size=100)
        elif motion_type == "constant_velocity":
            trajectory = constant_velocity_motion(self.T, self.dt, self.H, self.W, vx=10, vy=5, step_size=5)
        elif motion_type == "linear_acceleration":
            trajectory = linear_acceleration_motion(self.T, self.dt, self.H, self.W, acceleration=0.15, step_size=1)
        elif motion_type == "oscillatory":
            trajectory = oscillatory_motion(self.T, self.dt, self.H, self.W, amplitude=50, frequency=0.5, step_size=2)
        elif motion_type == "circular":
            trajectory = circular_motion(self.T, self.dt, self.H, self.W, radius=100, angular_velocity=1, step_size=2)
        elif motion_type == "simple_pendulum":
            trajectory = simple_pendulum_motion(self.T, self.dt, self.H, self.W, length=10, amplitude=0.5, gravity=-9.8, step_size=15)
        elif motion_type == "sine_wave_driven":
            trajectory = sine_wave_driven_motion(self.T, self.dt, self.H, self.W, amplitude=50, frequency=0.5, step_size=2)
        elif motion_type == "spiral":
            trajectory = spiral_motion(self.T, self.dt, self.H, self.W, radius=1, angular_velocity=1, expansion_rate=0.05, step_size=10)
        else:
            raise ValueError("Invalid motion type")
        return trajectory
    
    def generate(self, motion_type, noise_sigma: float = 0.01, max_num: int = 100):
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
        self.visualize_trajectory_with_trail(noisy_trajectory, f"{self.datadir}/{motion_type}.mp4")
        return True
        
    def visualize_trajectory_with_trail(self, trajectory, video_filename, max_trail_time=20, frame_rate=60):
        """
        Create a video of the trajectory with a trailing effect and save it to the file.
        
        Parameters:
        - trajectory: (T, 2) trajectory data, T is the number of frames, and 2 is the (x, y) coordinates per frame.
        - H: Height of the image (frame height).
        - W: Width of the image (frame width).
        - video_filename: Filename to save the video.
        - max_trail_time: Maximum time for the trail to persist, controlling the trail effect.
        - frame_rate: Video frame rate, controlling the frame rate of the generated video.
        """
        
        # Initialize video writer with lower quality settings (e.g., use 'MJPG' codec)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        # Using MJPG codec for faster encoding
        out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (self.W, self.H))
        
        # Create a white background for the frames
        background = np.ones((self.H, self.W, 3), dtype=np.uint8) * 255
        
        for t in range(len(trajectory)):
            # Copy the white background to prepare for drawing
            frame = background.copy()
            
            # Draw past points with decreasing opacity (trail effect)
            for i in range(t + 1):
                alpha = max(0, 1 - (t - i) / max_trail_time)  # Calculate alpha for transparency
                x, y = trajectory[i]
                x, y = int(x), int(y)
                
                if 0 < x < self.W-2 and 0 < y < self.H-2:
                    # Use alpha to make past points more transparent
                    overlay = frame[y-1:y+2, x-1:x+2]  # A small region around the point
                    overlay[:] = (int(overlay[0, 0, 0] * (1 - alpha)), 
                                int(overlay[0, 0, 1] * (1 - alpha)), 
                                int(overlay[0, 0, 2] * (1 - alpha)))
            
            # Draw the current point in black (completely opaque)
            x, y = trajectory[t]
            x, y = int(x), int(y)
            if 0 <= x < self.W and 0 <= y < self.H:
                cv2.circle(frame, (x, y), 3, (0, 0, 0), -1)  # Draw the current point

            # Write the current frame to the video file
            out.write(frame)
        
        # Release the video writer object
        out.release()
        print(f"Video saved as {video_filename}")


if __name__ == '__main__':
    np.random.seed(0)
    DATADIR = "data"
    T = 600  # Number of trajectory points
    W, H = 1280, 720  # Image dimensions (width, height)
    type_list = [
        "brownian_motion", 
        "constant_velocity", 
        "linear_acceleration", 
        "oscillatory", 
        "circular", 
        "simple_pendulum", 
        "sine_wave_driven", 
        "spiral",
    ]
    motion_generater = inanimate_generater(T, 1/60, H, W, DATADIR)
    for motion in type_list:
        print(motion)
        motion_generater.generate(motion, noise_sigma=0.01, max_num=100)