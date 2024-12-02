import os
import subprocess

def run_ffmpeg_command(command):
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def split_video(input_file, segment_time, output_pattern):
    """
    将视频按等长片段分割。

    :param input_file: 输入视频文件路径
    :param segment_time: 每个片段的时长，单位为秒
    :param output_pattern: 输出文件名的格式，比如 'output_%03d.mp4'
    """
    path = os.path.dirname(output_pattern)
    if path:
        os.makedirs(path, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c', 'copy',  # 不重新编码
        '-map', '0',  # 映射所有流
        '-segment_time', str(segment_time),  # 每个片段的时长
        '-f', 'segment',  # 使用 segment 多路输出格式
        '-reset_timestamps', '1',  # 每段重置时间戳
        output_pattern
    ]
    run_ffmpeg_command(command)

if __name__ == "__main__":
    input_file = "video/overcooked.mp4" # 替换为你的视频文件路径
    output_pattern = "video/overcooked/sample_%03d.mp4" # 输出文件名格式
    split_video(input_file, 10, output_pattern)