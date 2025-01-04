import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter


def vis(data):
    print(data.shape)
    print(np.max(data[:,[0,3]]),np.max(data[:,[1,4]]))

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