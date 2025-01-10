from moviepy.video.io.VideoFileClip import VideoFileClip

def cut_st_ed(input_video_path, output_folder, nameid):
    """
    :param input_video_path: 输入视频的路径（MP4格式）。
    :param output_folder: 输出样本存放的文件夹。
    默认切掉前5s和最后29秒
    """
    # 加载视频
    video = VideoFileClip(input_video_path)
    video_duration = video.duration  # 获取视频总时长（秒）
    
    
    start_time = 5
    end_time = video_duration - 29
    
    # 截取指定片段
    subclip = video.subclipped(start_time, end_time)
    output_path = f"{output_folder}/pro-ss-{nameid}.mp4"
    subclip.write_videofile(output_path, codec="libx264")
    print(f"导出样本: {output_path}")
    
    print("切完成！")

input_folder = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/supersmash-data/raw/"
output_folder = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/supersmash-data/processed/"

for i in range(6,56):
    input_path = f"{input_folder}/ss-{i}.mp4"
    cut_st_ed(input_path, output_folder, i-6)
