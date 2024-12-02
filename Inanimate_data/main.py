import os
import numpy as np
from generater import inanimate_generater


def single(data_num, data_dir):
    np.random.seed(0)
    T = 360  # Number of trajectory points
    W, H = 1280, 720  # Image dimensions (width, height)
    type_list = [
        "brownian_motion", 
        "constant_velocity", 
        "linear_acceleration",  
        "circular", 
        "simple_pendulum", 
        "sine_wave_driven", 
        "spiral",
    ]
    motion_generater = inanimate_generater(T, 1/60, H, W, data_dir)
    for motion in type_list:
        print(motion)
        os.makedirs(f"{data_dir}/{motion}", exist_ok=True)
        cnt = 0
        while cnt < data_num:
            noise_sigma = np.random.uniform(0, 5)
            data = motion_generater.generate(motion, noise_sigma, max_num=100)
            if data is False: continue
            else: 
                cnt += 1
                np.save(f"{data_dir}/{motion}/{cnt}.npy", data)


def double(data_num, data_dir):
    np.random.seed(0)
    T = 360  # Number of trajectory points
    W, H = 1280, 720  # Image dimensions (width, height)
    type_list = [
        "brownian_motion", 
        "constant_velocity", 
        "linear_acceleration",  
        "circular", 
        "simple_pendulum", 
        "sine_wave_driven", 
        "spiral",
    ]
    motion_generater = inanimate_generater(T, 1/60, H, W, data_dir)
    cnt = 0
    os.makedirs(f"{data_dir}/mixed", exist_ok=True)
    while cnt < data_num:
        # Randomly choose two motion types (can repeat)
        motion_type_1 = np.random.choice(type_list)
        motion_type_2 = np.random.choice(type_list)

        # Generate two separate trajectories
        noise_sigma_1 = np.random.uniform(0, 5)
        data_1 = motion_generater.generate(motion_type_1, noise_sigma_1, max_num=100)
        
        noise_sigma_2 = np.random.uniform(0, 5)
        data_2 = motion_generater.generate(motion_type_2, noise_sigma_2, max_num=100)
        
        # Check if data is successfully generated, if not continue
        if data_1 is False or data_2 is False:
            continue
        else:
            cnt += 1
            # Concatenate the two trajectories along axis 1 (features)
            combined_data = np.concatenate((data_1, data_2), axis=1)  # Shape becomes (T, 6)
            # Save the concatenated trajectory
            np.save(f"{data_dir}/mixed/{cnt}.npy", combined_data)
            motion_generater.vis_without_trail_2(combined_data, f"{data_dir}/videos/{cnt}.mp4")
            
            
if __name__ == '__main__':
    np.random.seed(0)
    
    data_num = 5
    data_dir = "data"
    os.makedirs(f"{data_dir}/videos", exist_ok=True)
    # single(data_num, DATADIR)
    double(data_num, data_dir)