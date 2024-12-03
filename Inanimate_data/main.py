import os
import numpy as np
from generater import inanimate_generater


def sample_single(motion_generater: inanimate_generater, motion_type: str, t: int, base = None):
    MAX = 100
    flag = False
    for i in range(MAX):
        data = motion_generater.generate(motion_type, t, base=base)
        if data is not False: 
            flag = True
            break
    if flag is False: 
        raise ValueError("Failed to generate a valid trajectory")
    return data

def double(motion_generater: inanimate_generater, type_list: list, t: int, base: list = [None, None]):
    motion_type_1 = np.random.choice(type_list)
    motion_type_2 = np.random.choice(type_list)
    data_1 = sample_single(motion_generater, motion_type_1, t, base[0])
    data_2 = sample_single(motion_generater, motion_type_2, t, base[1])
    return np.concatenate((data_1, data_2), axis=1)
            
def mixed(motion_generater: inanimate_generater, type_list: list, t: int):
    data_1 = double(motion_generater, type_list, int(t/2))
    base = [data_1[-1, 0:2], data_1[-1, 3:5]]
    data_2 = double(motion_generater, type_list, int(t/2), base)
    return np.concatenate((data_1, data_2), axis=0)
            
    
if __name__ == '__main__':
    np.random.seed(0)
    T = 360  # Number of trajectory points
    W, H = 1280, 720  # Image dimensions (width, height)
    data_num = 10
    data_dir = "data"
    os.makedirs(f"{data_dir}/motions", exist_ok=True)
    os.makedirs(f"{data_dir}/videos", exist_ok=True)
    type_list = [
        "brownian_motion", 
        "constant_velocity", 
        "linear_acceleration",  
        "circular", 
        "simple_pendulum", 
        "sine_wave_driven", 
        "spiral",
    ]
    motion_generater = inanimate_generater(1/60, H, W, data_dir)
    
    # Example usage
    # double(motion_generater, type_list, T)
    # mixed(motion_generater, type_list, T)
    
    for i in range(data_num):
        data = mixed(motion_generater, type_list, T)
        np.save(f"{data_dir}/motions/{i}.npy", data)
        # motion_generater.vis_without_trail_2(data, f"{data_dir}/videos/{i}.mp4")
    