import os

def seperate_npy_and_mp4(path):
    numpy_path = os.path.join(path, "numpy")
    os.makedirs(numpy_path, exist_ok=True)
    for file in os.listdir(path):
        if file.endswith(".npy"):
            os.rename(os.path.join(path, file), os.path.join(numpy_path, file))


if __name__ == "__main__":
    path = "data-trial/kof/video/output_slices_2/segmentation"
    seperate_npy_and_mp4(path)
