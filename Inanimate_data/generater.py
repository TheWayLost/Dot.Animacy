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
            acc = np.random.uniform(-1, 1) * 1.5
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
            radius_ = np.random.uniform(1, 5)
            angular_vel = np.random.uniform(-1, 1) * 10
            expansion_rate_ = np.random.uniform(0.2, 2)
            trajectory = spiral_motion(T, self.dt, self.H, self.W, radius=1, angular_velocity=angular_vel, expansion_rate=expansion_rate_, base=base)
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
        noise_sigma = np.random.uniform(0, 5)
        noisy_trajectory = noised_motion(trajectory, noise_sigma)
        r = sin_r(T, bias=25, w=10)
        noisy_trajectory = np.concatenate([noisy_trajectory, r[:, np.newaxis]], axis=1)
        # self.vis_without_trail_2(noisy_trajectory, f"{self.datadir}/{motion_type}.mp4")
        return noisy_trajectory
        
    def vis_as_video(self, trajectory, video_filename, frame_rate=60):
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

    def vis_as_image(self, trajectory, image_filename, skip_frames=30):
        """
        Create a trajectory visualization and save it as an image.

        Parameters:
        - trajectory: (T, 6) trajectory data, T is the number of frames, and 6 is the (x1, y1, r1, x2, y2, r2) coordinates per frame.
        - image_filename: Filename to save the image.
        - skip_frames: Number of frames to skip between each visualization step.
        """
        # Create a black background
        background = np.zeros((self.H, self.W, 4), dtype=np.uint8)  # Use 4 channels for RGBA

        # Define colors for the two points (with alpha channel)
        color_1 = (0, 0, 255, 255)  # Blue color for the first point (RGBA format)
        color_2 = (255, 0, 0, 255)  # Red color for the second point (RGBA format)

        # Normalize transparency across frames
        num_frames = len(trajectory)
        alpha_step = 255 / num_frames  # Transparency decrement per frame

        for t in range(0, num_frames, skip_frames):
            # Extract coordinates and radius for both points at time t
            x1, y1, r1, x2, y2, r2 = trajectory[t]

            # Convert to integer coordinates and radius
            x1, y1, r1 = int(x1), int(y1), int(r1)
            x2, y2, r2 = int(x2), int(y2), int(r2)

            # Compute transparency for the current frame
            alpha = int(255 - alpha_step * t)

            # Adjust colors with current alpha value
            color_1_with_alpha = (color_1[0], color_1[1], color_1[2], alpha)
            color_2_with_alpha = (color_2[0], color_2[1], color_2[2], alpha)

            # Draw the first point (in blue)
            if 0 <= x1 < self.W and 0 <= y1 < self.H:
                self.overlay_circle(background, (x1, y1), r1, color_1_with_alpha)

            # Draw the second point (in red)
            if 0 <= x2 < self.W and 0 <= y2 < self.H:
                self.overlay_circle(background, (x2, y2), r2, color_2_with_alpha)

        # Convert RGBA to RGB (removing transparency for saving as PNG)
        final_image = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)

        # Save the image
        cv2.imwrite(image_filename, final_image)
        print(f"Image saved as {image_filename}")

    def overlay_circle(self, image, center, radius, color):
        """
        Draw a circle with transparency on an RGBA image.

        Parameters:
        - image: The RGBA image.
        - center: (x, y) tuple for the circle center.
        - radius: Radius of the circle.
        - color: (R, G, B, A) color tuple with transparency.
        """
        overlay = image.copy()
        cv2.circle(overlay, center, radius, color[:3], -1)  # Draw circle on overlay

        # Blend the overlay with the original image based on alpha
        alpha = color[3] / 255.0
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def vis_trajectory_with_line(self, trajectory, image_filename, radius=10, skip_frames=1):
        """
        Visualize the trajectory with fixed radius for each point and fade effect from 0 to 1 for transparency.
        Also connect the points with a line.

        Parameters:
        - trajectory: (T, 6) trajectory data, T is the number of frames, and 6 is the (x1, y1, r1, x2, y2, r2) coordinates per frame.
        - image_filename: Filename to save the image.
        - radius: Fixed radius for the visualized points.
        - skip_frames: Number of frames to skip between each visualization step.
        """
        # Create a black background
        background = np.ones((self.H, self.W, 4), dtype=np.uint8)  # Use 4 channels for RGBA
        
        # Define colors for the two points (with alpha channel)
        color_1 = (0, 0, 255, 255)  # Blue color for the first point (RGBA format)
        color_2 = (255, 0, 0, 255)  # Red color for the second point (RGBA format)
        
        num_frames = len(trajectory)
        
        # Linear fade effect from 0 to 1 (transparency)
        for t in range(0, num_frames - 1, skip_frames):
            # Extract coordinates for both points at time t and t+1
            x1, y1, r1, x2, y2, r2 = trajectory[t]
            x1_next, y1_next, _, x2_next, y2_next, _ = trajectory[t + 1]
            
            # Convert to integer coordinates
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            x1_next, y1_next = int(x1_next), int(y1_next)
            x2_next, y2_next = int(x2_next), int(y2_next)
            
            # Compute transparency for the current frame (linear fade from 0 to 1)
            alpha = int(255 * (t / num_frames))  # From 0 to 255 based on frame index
            
            # Adjust colors with current alpha value
            color_1_with_alpha = (color_1[0], color_1[1], color_1[2], alpha)
            color_2_with_alpha = (color_2[0], color_2[1], color_2[2], alpha)
            
            # Draw the first point (in blue) with fixed radius
            if 0 <= x1 < self.W and 0 <= y1 < self.H:
                self.overlay_circle(background, (x1, y1), radius, color_1_with_alpha)
            
            # Draw the second point (in red) with fixed radius
            if 0 <= x2 < self.W and 0 <= y2 < self.H:
                self.overlay_circle(background, (x2, y2), radius, color_2_with_alpha)
            
            # # Draw line connecting points in the current and next frame
            # if 0 <= x1 < self.W and 0 <= y1 < self.H and 0 <= x1_next < self.W and 0 <= y1_next < self.H:
            #     line_color = (color_1[0], color_1[1], color_1[2], alpha)  # Line color with alpha
            #     cv2.line(background, (x1, y1), (x1_next, y1_next), line_color[:3], 2)
            
            # if 0 <= x2 < self.W and 0 <= y2 < self.H and 0 <= x2_next < self.W and 0 <= y2_next < self.H:
            #     line_color = (color_2[0], color_2[1], color_2[2], alpha)  # Line color with alpha
            #     cv2.line(background, (x2, y2), (x2_next, y2_next), line_color[:3], 2)

        # Convert RGBA to RGB (removing transparency for saving as PNG)
        final_image = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)
        
        # Save the image
        cv2.imwrite(image_filename, final_image)
        print(f"Image saved as {image_filename}")


