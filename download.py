import os
import subprocess

VIDEO_URL = "https://www.youtube.com/watch?v=FtutLA63Cp8"
FRAMES_DIR = "frames"
TARGETS_DIR = os.path.join(FRAMES_DIR, "targets")
OUTPUT_DIR = os.path.join(FRAMES_DIR, "output")
FPS = 15


def main():
    os.makedirs(TARGETS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(FRAMES_DIR, "bad_apple.mp4")
    audio_path = os.path.join(FRAMES_DIR, "bad_apple_audio.mp3")

    # Download video
    if not os.path.exists(video_path):
        print("Downloading Bad Apple video...")
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                "--merge-output-format",
                "mp4",
                "-o",
                video_path,
                VIDEO_URL,
            ],
            check=True,
        )
    else:
        print(f"Video already exists at {video_path}")

    # Extract audio
    if not os.path.exists(audio_path):
        print("Extracting audio...")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",
                audio_path,
            ],
            check=True,
        )
    else:
        print(f"Audio already exists at {audio_path}")

    # Extract frames at target FPS, resize to 256x256, grayscale
    existing = [f for f in os.listdir(TARGETS_DIR) if f.endswith(".png")]
    if len(existing) < 10:
        print(f"Extracting frames at {FPS}fps, 256x256, grayscale...")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"fps={FPS},scale=256:256,format=gray",
                os.path.join(TARGETS_DIR, "frame_%05d.png"),
            ],
            check=True,
        )
        count = len([f for f in os.listdir(TARGETS_DIR) if f.endswith(".png")])
        print(f"Extracted {count} frames")
    else:
        print(f"Frames already extracted ({len(existing)} found)")


if __name__ == "__main__":
    main()
