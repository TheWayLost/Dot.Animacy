import os
import numpy as np
from generater import inanimate_generater
from tqdm import tqdm
from utils import visulizer

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
      
def interp(x):
    """
    Generate a tensor y of shape (2N-1, C) from input tensor x of shape (N, C).
    Even indices of y are the rows of x, and odd indices are the linear interpolation 
    between adjacent rows of x.

    :param x: torch.Tensor, input tensor of shape (N, C)
    :return: torch.Tensor, output tensor of shape (2N-1, C)
    """
    N, T, C = x.shape
    y = np.empty((N, 2 * T - 1, C), dtype=x.dtype)
    y[:, 0::2] = x
    y[:, 1::2] = (x[:, :-1] + x[:, 1:]) / 2
    
    return y
          
    
if __name__ == '__main__':
    np.random.seed(0)
    FPS = 30
    SECONDS = 15
    T = FPS * SECONDS  # Number of trajectory points
    W, H = 1280, 720  # Image dimensions (width, height)
    visulizer = visulizer(W, H, FPS, SECONDS)
    
    data_num = 0
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
    data = np.load(f"tmp/tmp.npy")
    data = data[-6:]
    data = interp(data)
    data = interp(data)
    for i in range(6):
        tmp_data = data[i]
        print(tmp_data.shape)
        visulizer.vis_trajectory_with_line(tmp_data, f"tmp/{i}_0.png", radius=6, skip_frames=1)
    
    if data_num > 0:
        dataset = []
        for i in tqdm(range(data_num)):
            if np.random.rand() < 0.0: # ban this mode
                data = mixed(motion_generater, type_list, T) # noqa
            else:
                data = double(motion_generater, type_list, T)
            dataset.append(data)
            np.save(f"{data_dir}/motions/{i}.npy", data)
            visulizer.vis_as_video(data, f"{data_dir}/videos/{i}.mp4", frame_rate=FPS)
            # visulizer.vis_as_image(data, f"{data_dir}/videos/{i}.png", skip_frames=5)
            visulizer.vis_trajectory_with_line(data, f"{data_dir}/videos/{i}_1.png", radius=3, skip_frames=1)
        inanimacy_dataset = np.array(dataset)
        print(inanimacy_dataset.shape)
        np.save(f"{data_dir}/inanimacy_dataset.npy", inanimacy_dataset)
    