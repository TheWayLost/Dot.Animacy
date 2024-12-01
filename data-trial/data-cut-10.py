from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_video_path, output_folder, sample_duration=10):
    """
    将一个长视频切分成指定时长的样本。
    
    :param input_video_path: 输入视频的路径（MP4格式）。
    :param output_folder: 输出样本存放的文件夹。
    :param sample_duration: 每段视频的时长（秒），默认10秒。
    """
    # 加载视频
    video = VideoFileClip(input_video_path)
    video_duration = video.duration  # 获取视频总时长（秒）
    
    # 计算切分数量
    num_clips = int(video_duration // sample_duration)
    
    # 切分视频
    for i in range(num_clips):
        start_time = i * sample_duration
        end_time = start_time + sample_duration
        
        # 截取指定片段
        subclip = video.subclipped(start_time, end_time)
        output_path = f"{output_folder}/sample_{i+1}.mp4"
        subclip.write_videofile(output_path, codec="libx264")
        print(f"导出样本: {output_path}")
    
    # 检查是否还有剩余部分未切分
    if video_duration % sample_duration != 0:
        start_time = num_clips * sample_duration
        subclip = video.subclipped(start_time, video_duration)
        output_path = f"{output_folder}/sample_{num_clips+1}.mp4"
        subclip.write_videofile(output_path, codec="libx264")
        print(f"导出剩余样本: {output_path}")
    
    print("切分完成！")


# 使用示例
input_video_path = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/data-trial/SuperSmash/kirby-joker-1.mp4"  # 替换视频路径
output_folder = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/data-trial/SuperSmash/SuperSmash-1"    # 替换为存放切分视频的文件夹路径
split_video(input_video_path, output_folder)