import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter


kof_path = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/kof_data/segmentation_split/"

def vis(data):
    print(data.shape)
    print(np.max(data[:,[0,3]]),np.max(data[:,[1,4]]))

    xaxis = np.arange(len(data))
    r1 = data[:,2].flatten()
    r2 = data[:,5].flatten()
    sr1 = savgol_filter(r1, window_length=15, polyorder=2)
    sr2 = savgol_filter(r2, window_length=15, polyorder=2)
    plt.plot(xaxis, r1, label="blue",color="blue")
    plt.plot(xaxis, r2, label='red', color='red')
    plt.plot(xaxis, sr1, label="blue-s",color="green")
    plt.plot(xaxis, sr2, label='red-s', color='orange')
    plt.legend()
    plt.show()

    x1 = data[:,0].flatten()
    x2 = data[:,3].flatten()
    sx1 = savgol_filter(x1, window_length=15, polyorder=2)
    sx2 = savgol_filter(x2, window_length=15, polyorder=2)
    plt.plot(xaxis, x1, label="blue",color="blue")
    plt.plot(xaxis, x2, label='red', color='red')
    plt.plot(xaxis, sx1, label="blue-s",color="green")
    plt.plot(xaxis, sx2, label='red-s', color='orange')
    plt.legend()
    plt.show()

    y1 = data[:,1].flatten()
    y2 = data[:,4].flatten()
    sy1 = savgol_filter(y1, window_length=15, polyorder=2)
    sy2 = savgol_filter(y2, window_length=15, polyorder=2)
    plt.plot(xaxis, y1, label="blue",color="blue")
    plt.plot(xaxis, y2, label='red', color='red')
    plt.plot(xaxis, sy1, label="blue-s",color="green")
    plt.plot(xaxis, sy2, label='red-s', color='orange')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(12.8, 7.2))  # 设置显示图像的大小为1280x720
    for i in range(0,len(data),2):
        ax.clear()
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        circle1 = patches.Circle((sx1[i], sy1[i]), radius=sr1[i], color='blue', alpha=0.5)
        ax.add_patch(circle1)
        circle2 = patches.Circle((sx2[i], sy2[i]), radius=sr2[i], color='red', alpha=0.5)
        ax.add_patch(circle2)
        plt.pause(0.066) 
    plt.show()



def smo(npy_path):
    data = np.load(npy_path)
    print(data.shape)
    print(np.max(data[:,[0,3]]),np.max(data[:,[1,4]]))

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



# vis(np.load(kof_path + str(17)+".npy"))