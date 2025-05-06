import subprocess
import os


def video_to_frames(video_path):
    output_folder = 'static/DLFile/videoPic/uploadFrames'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', 'fps=30',
        os.path.join(output_folder, 'frame_%04d.png')
    ]
    subprocess.run(command, check=True)


def frames_to_video(input_folder, fps=30):
    output_video_path = 'static/DLFile/output_folder/'
    if not output_video_path.endswith('.mp4'):
        output_video_path += 'result.mp4'
    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(input_folder, 'output_frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    subprocess.run(command, check=True)

    return output_video_path