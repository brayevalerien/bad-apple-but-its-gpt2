import os
import subprocess
import sys

FRAMES_DIR = "frames"
OUTPUT_DIR = os.path.join(FRAMES_DIR, "output")
AUDIO_PATH = os.path.join(FRAMES_DIR, "bad_apple_audio.mp3")
FPS = 15


def main():
    # Check frames exist
    frame_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png"))
    frame_files = [f for f in frame_files if not os.path.isdir(os.path.join(OUTPUT_DIR, f))]
    if not frame_files:
        print(f"No frames found in {OUTPUT_DIR}")
        sys.exit(1)
    print(f"Found {len(frame_files)} frames")

    # Assemble video without audio
    video_noaudio = "output_noaudio.mp4"
    print("Assembling video...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            os.path.join(OUTPUT_DIR, "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            video_noaudio,
        ],
        check=True,
    )
    print(f"Video assembled: {video_noaudio}")

    # Mux audio
    if os.path.exists(AUDIO_PATH):
        final_output = "bad_apple_attention.mp4"
        print("Muxing audio...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_noaudio,
                "-i",
                AUDIO_PATH,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                final_output,
            ],
            check=True,
        )
        os.remove(video_noaudio)
        print(f"Final video: {final_output}")
    else:
        print(f"Audio not found at {AUDIO_PATH}, video saved without audio as {video_noaudio}")
        os.rename(video_noaudio, "bad_apple_attention.mp4")


if __name__ == "__main__":
    main()
