from time import perf_counter
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

# 替换为你的 MP4 文件路径
video_path = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/data-trial/SuperSmash/SuperSmash-1/sample-12.mp4"

# 使用对应的配置文件和模型检查点
cfg, ckpt = "sam2.1_hiera_t.yaml", "/home/laura5ia/sam2/checkpoints/sam2.1_hiera_tiny.pt"


# 如果有 GPU 可用，使用 GPU，否则使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 构建视频预测器
predictor = build_sam2_video_predictor(cfg, ckpt, device)
inference_state = predictor.init_state(
    video_path=video_path,
    async_loading_frames=True
)

print(")0")

predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[210, 350]], dtype=np.float32),
    labels=np.array([1], np.int32),
)

tprev = -1
for result in predictor.propagate_in_video(inference_state):
    # Do nothing with results, just report VRAM use
    if  (perf_counter() > tprev + 1.0) and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print("VRAM:", (total_bytes - free_bytes) // 1_000_000, "MB")
        tprev = perf_counter()
    pass

print("HERE!")