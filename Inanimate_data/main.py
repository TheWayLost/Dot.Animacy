import os
import numpy as np
from generater import inanimate_generater
from tqdm import tqdm

def sample_single(motion_generater: inanimate_generater, motion_type: str, t: int, base = None):
    MAX = 500
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
    T = 180  # Number of trajectory points
    FPS = 30
    W, H = 1280, 720  # Image dimensions (width, height)
    data_num = 20
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
    motion_generater = inanimate_generater(1/FPS, H, W, data_dir)
    
    # Example usage
    # double(motion_generater, type_list, T)
    # mixed(motion_generater, type_list, T)
    # for i in range(2, 12):
    #     data = np.load(f"tmp/{i}.npy")
    #     print(data.shape)
    #     motion_generater.vis_trajectory_with_line(data, f"tmp/{i}_0.png", radius=6, skip_frames=1)
    
    if data_num > 0:
        dataset = []
        for i in tqdm(range(data_num)):
            if np.random.rand() < 0.0: # ban this mode
                data = mixed(motion_generater, type_list, T) # noqa
            else:
                data = double(motion_generater, type_list, T)
            dataset.append(data)
            np.save(f"{data_dir}/motions/{i}.npy", data)
            motion_generater.vis_as_video(data, f"{data_dir}/videos/{i}.mp4")
            # motion_generater.vis_as_image(data, f"{data_dir}/videos/{i}.png", skip_frames=5)
            motion_generater.vis_trajectory_with_line(data, f"{data_dir}/videos/{i}_1.png", radius=3, skip_frames=1)
        inanimacy_dataset = np.array(dataset)
        print(inanimacy_dataset.shape)
        np.save(f"{data_dir}/inanimacy_dataset.npy", inanimacy_dataset)
    