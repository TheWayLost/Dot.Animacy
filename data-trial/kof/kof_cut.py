import math
import os

import cv2
import numpy as np
from tqdm import tqdm


def is_nearly_black(frame, frame_height,threshold=10):
    return np.mean(frame[:frame_height, :, :]) < threshold


def split_video_by_black_frames(video_path, output_dir, offset=0):
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
    video_idx = offset+1
    os.makedirs(output_dir, exist_ok=True)
    for i, end_frame in enumerate(tqdm(black_frame_indices)):
        if end_frame > start_frame:
            if end_frame > start_frame+fps*10:
                cut_video(video_path, start_frame, end_frame, output_dir, video_idx)
                video_idx += 1
            start_frame = end_frame

    # 处理最后一段
    if start_frame+fps*10 < frame_count:
        cut_video(video_path, start_frame, frame_count, output_dir, video_idx)


def cut_video(video_path, start_frame, end_frame, output_dir, idx):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.join(output_dir, "snapshot"), exist_ok=True)
    output_path = f"{output_dir}/slice_{idx}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        # 特别地，将第一帧保存为同名的jpg文件
        if frame_idx == start_frame:
            cv2.imwrite(f"{output_dir}/snapshot/slice_{idx}.jpg", frame)
        if not ret:
            break
        out.write(frame)
    out.release()
    cap.release()


# 使用示例
offset = 0
split_video_by_black_frames("video/input_2.mp4", "video/output_slices_2",57)