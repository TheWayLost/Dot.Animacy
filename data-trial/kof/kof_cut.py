import math
import os

import cv2
import numpy as np
from tqdm import tqdm


def is_nearly_black(frame, frame_height,threshold=10):
    return np.mean(frame[:frame_height, :, :]) < threshold


def split_video_by_black_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    # 获得帧的高宽
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    logo_height = math.ceil(frame_height*0.0324)
    black_frame_indices = []

    with tqdm(total=frame_count, desc="Processing Frames", unit="frame") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if is_nearly_black(frame, int(frame_height*0.1)):
                black_frame_indices.append(frame_idx)
            frame_idx += 1
            pbar.update(1)
    cap.release()

    # 按照纯黑帧分割视频
    start_frame = 0
    video_idx = 0
    os.makedirs(output_dir, exist_ok=True)
    for i, end_frame in enumerate(tqdm(black_frame_indices)):
        if end_frame > start_frame:
            if end_frame > start_frame+fps*10:
                cut_video(video_path, start_frame, end_frame, output_dir, video_idx)
                video_idx += 1
            start_frame = end_frame

    # 处理最后一段
    if start_frame < frame_count:
        cut_video(video_path, start_frame, frame_count, output_dir, len(black_frame_indices))


def cut_video(video_path, start_frame, end_frame, output_dir, idx):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = f"{output_dir}/slice_{idx}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    cap.release()


# 使用示例
split_video_by_black_frames("input.mp4", "output_slices")