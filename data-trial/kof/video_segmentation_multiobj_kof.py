#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults


# Define pathing & device usage
def trajectory_extract(video_path, model_path, output_video_path, prompts_per_frame_index,
        imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True},
        color_dict = {
        "obj1": (0, 0, 255),  # 红色
        "obj2": (255, 0, 0),  # 蓝色
    }):
    device, dtype = "cpu", torch.float32
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    print("device: ",device)
    # For demo purposes, we'll define all prompts ahead of time and store them per frame index & object
    # -> First level key (e.g. 0, 30, 35) represents the frame index where the prompts should be applied
    # -> Second level key (e.g. 'obj1', 'obj2') represents which 'object' the prompt belongs to for tracking purposes
    enable_prompt_visualization = True
    # *** These prompts are set up for a video of horses available from pexels.com ***
    # https://www.pexels.com/video/horses-running-on-grassland-4215784/
    # By: Adrian Hoparda

    # Set up memory storage for tracked objects
    # -> Assumes each object is represented by a unique dictionary key (e.g. 'obj1')
    # -> This holds both the 'prompt' & 'recent' memory data needed for tracking!
    memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

    # Read first frame to check that we can read from the video, then reset playback
    vcap = cv2.VideoCapture(video_path)
    ok_frame, first_frame = vcap.read()
    if not ok_frame:
        raise IOError(f"Unable to read video frames: {video_path}")
    vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 定义输出视频路径和编码格式
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = int(vcap.get(cv2.CAP_PROP_FPS))  # 获取输入视频的帧率
    print(fps)
    frame_height, frame_width = first_frame.shape[:2]
    output_size = (frame_width, frame_height)  # sidebyside_frame 的尺寸

    # 初始化 VideoWriter 对象
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, output_size, True)
    # 检查 VideoWriter 是否成功打开
    if not video_writer.isOpened():
        raise IOError(f"Failed to create video file at {output_video_path}")

    # Set up model
    print("Loading model...")
    model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
    sammodel.to(device=device, dtype=dtype)

    # radius_scaling =1

    # Process video frames
    close_keycodes = {27, ord("q")}  # Esc or q to close
    try:
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        feature_list = [] # x_1, y_1, r_1, x_2, y_2, r_2
        for frame_idx in tqdm(range(total_frames)): 

            # Read frames
            ok_frame, frame = vcap.read()
            if not ok_frame:
                print("", "Done! No more frames...", sep="\n")
                break

            # Encode frame data (shared for all objects)
            encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

            # Generate & store prompt memory encodings for each object as needed
            prompts_dict = prompts_per_frame_index.get(frame_idx, None)
            if prompts_dict is not None:

                # Loop over all sets of prompts for the current frame
                for obj_key_name, obj_prompts in prompts_dict.items():
                    print(f"Generating prompt for object: {obj_key_name} (frame {frame_idx})")
                    init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts)
                    memory_per_obj_dict[obj_key_name].store_prompt_result(frame_idx, init_mem, init_ptr)

                    # Draw prompts for debugging
                    if enable_prompt_visualization:
                        prompt_vis_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                        norm_to_px_factor = np.float32((prompt_vis_frame.shape[1] - 1, prompt_vis_frame.shape[0] - 1))
                        for xy_norm in obj_prompts.get("fg_xy_norm_list", []):
                            xy_px = np.int32(xy_norm * norm_to_px_factor)
                            cv2.circle(prompt_vis_frame, xy_px, 3, (0, 255, 0), -1)
                        for xy_norm in obj_prompts.get("bg_xy_norm_list", []):
                            xy_px = np.int32(xy_norm * norm_to_px_factor)
                            cv2.circle(prompt_vis_frame, xy_px, 3, (0, 0, 255), -1)
                        for xy1_norm, xy2_norm in obj_prompts.get("box_tlbr_norm_list", []):
                            xy1_px = np.int32(xy1_norm * norm_to_px_factor)
                            xy2_px = np.int32(xy2_norm * norm_to_px_factor)
                            cv2.rectangle(prompt_vis_frame, xy1_px, xy2_px, (0, 255, 255), 2)

                        # Show prompt in it's own window and close after viewing
                        wintitle = f"Prompt ({obj_key_name}) - Press key to continue"
                        cv2.imshow(wintitle, prompt_vis_frame)
                        cv2.waitKey(0)
                        cv2.destroyWindow(wintitle)

            # Update tracking using newest frame
            # combined_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
            combined_mask_result = np.zeros((*frame.shape[:2], 3), dtype=np.uint8) # 颜色
            mask_result = np.zeros((*frame.shape[:2], 3), dtype=np.uint8) # 颜色
            feature = []
            for obj_key_name, obj_memory in memory_per_obj_dict.items():
                obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                    encoded_imgs_list, **obj_memory.to_dict()
                )

                # Skip storage for bad results (often due to occlusion)
                obj_score = obj_score.item()
                if obj_score < 0:
                    print(f"Bad object score for {obj_key_name}! Skipping memory storage...")
                    continue

                # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
                # -> This can be commented out and tracking may still work, if object doesn't change much
                obj_memory.store_result(frame_idx, mem_enc, obj_ptr)

                # Add object mask prediction to 'combine' mask for display
                # -> This is just for visualization, not needed for tracking
                obj_mask = torch.nn.functional.interpolate(
                    mask_preds[:, best_mask_idx, :, :],
                    size=combined_mask_result.shape[0:2],
                    mode="bilinear",
                    align_corners=False,
                )
                obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
                mask_result = np.bitwise_or(mask_result, np.stack([obj_mask_binary] * 3, axis = -1))

                # draw dot (circle)
                mask_coords = np.argwhere(obj_mask_binary)
                if mask_coords.size > 0:
                    center_y, center_x = mask_coords.mean(axis=0).astype(int)  # 质心
                    area = mask_coords.shape[0]  # 掩码面积
                    radius = int(np.sqrt(area / np.pi)/1.1)  # 根据面积计算半径
                    if frame_idx == 0:
                        print(radius)
                    # 创建一个全零数组，大小和 obj_mask_binary 一样
                    circular_mask = np.zeros_like(combined_mask_result, dtype=np.uint8)
                    color = color_dict.get(obj_key_name, (255, 255, 255))  # 默认白色
                if obj_key_name == "obj1" or obj_key_name == "obj2":
                    feature.extend([center_x, center_y, radius])
                cv2.circle(circular_mask, (center_x, center_y), radius, color, -1)  # 圆内部像素设置为 1
                obj_mask_binary = (circular_mask > 0.0).squeeze()
                if obj_key_name == "obj3":
                    obj_mask_binary = np.stack([(obj_mask > 0.0).cpu().numpy().squeeze()] * 3, axis = -1)


                combined_mask_result = np.bitwise_or(combined_mask_result, obj_mask_binary)
            feature_list.append(np.array(feature))
            # Combine original image & mask result side-by-side for display
            combined_mask_result_uint8 = combined_mask_result.astype(np.uint8) * 255
            mask_result_uint8 = mask_result.astype(np.uint8) * 255
            gray_mask = mask_result_uint8
            # disp_mask = cv2.cvtColor(mask_result_uint8, cv2.COLOR_GRAY2BGR)
            disp_mask = combined_mask_result_uint8
            sidebyside_frame = np.hstack((frame, disp_mask, gray_mask))
            sidebyside_frame = cv2.resize(sidebyside_frame, dsize=None, fx=0.5, fy=0.5)

            # 写入帧到输出视频
            # print(disp_mask.shape)
            video_writer.write(disp_mask)
            # Show result
            cv2.imshow("Video Segmentation Result - q to quit", sidebyside_frame)

            #print(disp_mask.shape, output_size)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress in close_keycodes:
                break
        np.save(output_video_path.replace(".mp4", ".npy"), np.array(feature_list))
    except Exception as err:
        raise err

    except KeyboardInterrupt:
        print("Closed by ctrl+c!")

    finally:
        vcap.release()
        video_writer.release()  # 释放 VideoWriter
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video/output_slices/slice_82.mp4"  # 替换为实际视频路径
    model_path = "model_weights/sam2.1_hiera_tiny.pt"  # 替换为实陼模型路径
    output_video_path = "video/output_slices/out_35.mp4"  # 替换为实际保存路径
    prompts_per_frame_index = {  ## 可以改变起始位置
        0: {
            "obj1": {
                "box_tlbr_norm_list": [],
                "fg_xy_norm_list": [(74/320, 109/180)],
                "bg_xy_norm_list": [],
            },
            "obj2": {
                "box_tlbr_norm_list": [],
                "fg_xy_norm_list": [(246/320,109/180)],
                "bg_xy_norm_list": [],
            },
        }
    }
    trajectory_extract(video_path, model_path, output_video_path, prompts_per_frame_index)