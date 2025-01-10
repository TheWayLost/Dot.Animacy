import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw


kof_path = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/kof_data/segmentation_split/"

def vis(data):
    print(data.shape)
    print(np.max(data[:,[0,3]]),np.max(data[:,[1,4]]),np.max(data[:,[2,5]]),np.min(data[:,[2,5]]))

    fig, ax = plt.subplots(2, 2,figsize=(12.8, 7.2))

    xaxis = np.arange(len(data))
    r1 = data[:,2].flatten()
    r2 = data[:,5].flatten()
    sr1 = savgol_filter(r1, window_length=15, polyorder=2)
    sr2 = savgol_filter(r2, window_length=15, polyorder=2)
    ax[0,0].plot(xaxis, r1, label="blue",color="blue")
    ax[0,0].plot(xaxis, r2, label='red', color='red')
    ax[0,0].plot(xaxis, sr1, label="blue-s",color="green")
    ax[0,0].plot(xaxis, sr2, label='red-s', color='orange')
    ax[0,0].legend()
    ax[0,0].set_title("radius")

    x1 = data[:,0].flatten()
    x2 = data[:,3].flatten()
    sx1 = savgol_filter(x1, window_length=15, polyorder=2)
    sx2 = savgol_filter(x2, window_length=15, polyorder=2)
    ax[0,1].plot(xaxis, x1, label="blue",color="blue")
    ax[0,1].plot(xaxis, x2, label='red', color='red')
    ax[0,1].plot(xaxis, sx1, label="blue-s",color="green")
    ax[0,1].plot(xaxis, sx2, label='red-s', color='orange')
    ax[0,1].legend()
    ax[0,1].set_title("x")

    y1 = data[:,1].flatten()
    y2 = data[:,4].flatten()
    sy1 = savgol_filter(y1, window_length=15, polyorder=2)
    sy2 = savgol_filter(y2, window_length=15, polyorder=2)
    ax[1,0].plot(xaxis, y1, label="blue",color="blue")
    ax[1,0].plot(xaxis, y2, label='red', color='red')
    ax[1,0].plot(xaxis, sy1, label="blue-s",color="green")
    ax[1,0].plot(xaxis, sy2, label='red-s', color='orange')
    ax[1,0].legend()
    ax[1,0].set_title("y")

    ax[1,1].set_title("video clip")

    for i in range(0,len(data),2):
        ax[1,1].clear()
        ax[1,1].set_xlim(0, 1280)
        ax[1,1].set_ylim(0, 720)
        circle1 = patches.Circle((sx1[i], sy1[i]), radius=sr1[i], color='blue', alpha=0.5)
        ax[1,1].add_patch(circle1)
        circle2 = patches.Circle((sx2[i], sy2[i]), radius=sr2[i], color='red', alpha=0.5)
        ax[1,1].add_patch(circle2)
        plt.pause(0.07) 
    plt.show()



def smo(k):
    npy_path=kof_path + str(k)+".npy"
    data = np.load(npy_path)
    # print(data.shape)
    # print(np.max(data[:,[0,3]]),np.max(data[:,[1,4]]))

    xaxis = np.arange(len(data))
    r1 = data[:,2].flatten()
    r2 = data[:,5].flatten()
    sr1 = savgol_filter(r1, window_length=15, polyorder=2)
    sr2 = savgol_filter(r2, window_length=15, polyorder=2)

    x1 = data[:,0].flatten()
    x2 = data[:,3].flatten()
    sx1 = savgol_filter(x1, window_length=15, polyorder=2)
    sx2 = savgol_filter(x2, window_length=15, polyorder=2)

    y1 = data[:,1].flatten()
    y2 = data[:,4].flatten()
    sy1 = savgol_filter(y1, window_length=15, polyorder=2)
    sy2 = savgol_filter(y2, window_length=15, polyorder=2)

    result = np.column_stack((sx1, sy1, sr1, sx2, sy2, sr2))
    np.save("kof-smo-"+str(k)+".npy",result)


def lossvis(file_path):
    epochs = []
    losses = []
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         match = re.match(r"Epoch (\d+): \d+it.*loss=(\S+)]", line)
    #         if match:
    #             epoch = int(match.group(1))
    #             loss = float(match.group(2))
# 
    #             epochs.append(epoch)
    #             losses.append(loss)
    losslog = np.load(file_path)
    losses = losslog
    epochs = range(losslog.shape[0])
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, linestyle='-', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()



def create_circle_video(data, background_color="white", output_path="circle_vis-gt2.mp4", fps=30, width=1280, height=720):
    """
    input data的 size 是: (anything, 6)
    
    """
    r1 = data[:,2].flatten()
    r2 = data[:,5].flatten()
    sr1 = savgol_filter(r1, window_length=15, polyorder=2)
    sr2 = savgol_filter(r2, window_length=15, polyorder=2)

    x1 = data[:,0].flatten()
    x2 = data[:,3].flatten()
    sx1 = savgol_filter(x1, window_length=15, polyorder=2)
    sx2 = savgol_filter(x2, window_length=15, polyorder=2)

    y1 = data[:,1].flatten()
    y2 = data[:,4].flatten()
    sy1 = savgol_filter(y1, window_length=15, polyorder=2)
    sy2 = savgol_filter(y2, window_length=15, polyorder=2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def create_frame(x1, y1, r1, x2, y2, r2):
        """Creates a single frame with the two circles."""
        img = Image.new("RGB", (width, height), color=background_color)
        draw = ImageDraw.Draw(img)
        y1 = 720 - y1
        y2 = 720 - y2
        if r1 < 0: r1 = 0.1
        if r2 < 0: r2 = 0.1

        circle2 = (x1 - r1, y1 - r1, x1 + r1, y1 + r1)
        circle1 = (x2 - r2, y2 - r2, x2 + r2, y2 + r2)

        if r1 >= r2:
            draw.ellipse(circle2, fill="blue")
            draw.ellipse(circle1, fill="red")
        else:
            draw.ellipse(circle1, fill="red")
            draw.ellipse(circle2, fill="blue")

        return np.array(img) # Convert PIL image to numpy array

    for i in range(data.shape[0]):
        frame = create_frame(sx1[i],sy1[i],sr1[i],sx2[i],sy2[i],sr2[i])
        # Convert from RGB to BGR format for cv2:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"Video saved to {output_path}")


def pathVis(data):
    selected_data = data[:, 8]  # Shape becomes (6, 90, 6)

    # Loop over all 90 circles to visualize their paths
    num_circles = selected_data.shape[1]  # 90 circles
    steps = selected_data.shape[0]  # 6 steps

    # Create a plot
    plt.figure(figsize=(10, 10))

    # Iterate over all circles to plot their paths
    for circle_idx in range(num_circles):
        # Extract the path of circle `circle_idx` for (x1, y1) and (x2, y2)
        x1_path = selected_data[:, circle_idx, 0]  # x1 over 6 steps
        y1_path = selected_data[:, circle_idx, 1]  # y1 over 6 steps
        x2_path = selected_data[:, circle_idx, 3]  # x2 over 6 steps
        y2_path = selected_data[:, circle_idx, 4]  # y2 over 6 steps

        # Plot the paths for the two circles
        plt.plot(x1_path, y1_path, 'r-o', label=f'Circle 1 (ID: {circle_idx})' if circle_idx == 0 else "")  # Circle 1
        plt.plot(x2_path, y2_path, 'b-o', label=f'Circle 2 (ID: {circle_idx})' if circle_idx == 0 else "")  # Circle 2

        # Optionally mark the start and end points
        plt.scatter(x1_path[0], y1_path[0], c='red', marker='x')  # Start of Circle 1
        plt.scatter(x2_path[0], y2_path[0], c='blue', marker='x')  # Start of Circle 2

    # Set plot details
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Paths of Two Circles Over 6 Steps")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


def pathVis3d(data):

    def map_to_range(values, new_min, new_max):
        old_min, old_max = values.min(), values.max()
        return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    selected_data = data[:, 8]  # 变为 (6, 90, 6)

    # 创建三维图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 时间步
    timesteps = np.linspace(0, 1, selected_data.shape[0])  # 6 个时间步

    # 遍历所有圆的路径
    for timestep_idx in range(selected_data.shape[0]):  # Iterate over 6 timesteps
        # Extract all (x1, y1) for this timestep
        x1_positions = selected_data[timestep_idx, :, 0]  # All x1 at this timestep
        y1_positions = selected_data[timestep_idx, :, 1]  # All y1 at this timestep
        if timestep_idx == 0:
            x1_positions = map_to_range(x1_positions, 0, 1000)
            y1_positions = map_to_range(y1_positions, 0, 600)
        z_positions = [timesteps[timestep_idx]] * len(x1_positions)  # Fixed z (time)

        # Scatter plot the points
        ax.scatter(x1_positions, y1_positions, z_positions, c='r', marker='.', label=f"Timestep {timestep_idx}" if timestep_idx == 0 else "")

        # Draw lines connecting the points sequentially
        for i in range(len(x1_positions) - 1):  # Sequential links
            ax.plot(
                [x1_positions[i], x1_positions[i + 1]],
                [y1_positions[i], y1_positions[i + 1]],
                [z_positions[i], z_positions[i + 1]],
                'r-'  # Black line
            )

    # Set axis labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Time')
    ax.set_title('Sequentially Linked (x1, y1) Points at Each Timestep')

    plt.show()


def modelVis(ckpt):
    pass # TODO


if __name__ == "__main__":
    # lossfile_path = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/SimpleConv1d_400e_loss_norm_smoData_swap.npy"
    # lossvis(lossfile_path)

    # res = np.load("/mnt/d/my_github/dotAnimacy/Dot.Animacy/SimpleConv1d_cond_222e_10sample.npy")
    # vis(res[9].repeat(2,0))
    # for i in range(10):
    #     vis(res[i].repeat(2,0))
    # vis(np.load(kof_path + str(1)+".npy"))

    #数据smooth
    #for i in tqdm(range(785)):
    #    smo(i)

    # vis(res[8].repeat(2,0))
    # print(res[8][[0,44,89],:])
    # create_circle_video(res[8].repeat(2,0), output_path="circle_vis-simpleCond-timeemb.mp4")

    res = np.load("/mnt/d/my_github/dotAnimacy/Dot.Animacy/SimpleConv1d_500epoch_path_smoData_swap.npy")
    print(res.shape)
    # #pathVis(res)
    # #pathVis3d(res)
    create_circle_video(res[5][8].repeat(2,0), output_path="circle_vis-swap-8-6sec.mp4")
    # #for i in range(6):
    # #    vis(res[i][8].repeat(2,0))

    #res = np.load("/mnt/d/my_github/dotAnimacy/Dot.Animacy/kof_data/smoothed/kof-smo-444.npy")
    #  print(res.shape)
    #create_circle_video(res[::2,:].repeat(2,0), output_path="circle_vis-gt-smo-444.mp4")


