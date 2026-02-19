import os
import subprocess

def play_video(video_path):

    if not os.path.exists(video_path):
        print("Video not found:", video_path)
        return

    # Open VLC as external app (cleanest solution on macOS)
    subprocess.run([
        "/Applications/VLC.app/Contents/MacOS/VLC",
        "--play-and-exit",
        video_path
    ])
