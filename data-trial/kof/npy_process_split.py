import os
import numpy as np

def process_and_split(data_array:np.ndarray,frame_num = 180):
    for i in range(1,data_array.shape[0]):
        data_array[i] = np.where(data_array[i] == 0, data_array[i-1], data_array[i])
    slice_num = data_array.shape[0]//frame_num
    return [data_array[i*frame_num:(i+1)*frame_num] for i in range(slice_num)]

def process_dir(input_dir,output_dir):
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir should not be the same")
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            data_array = np.load(f"{input_dir}/{file}")
            slices = process_and_split(data_array)
            for slice in slices:
                np.save(f"{output_dir}/{idx}.npy",slice)
                idx += 1


if __name__ == "__main__":
    input_dir = "data-trial/kof/video/output_slices_1/numpy"
    output_dir = "data-trial/kof/video/segmentation_split"
    process_dir(input_dir,output_dir)