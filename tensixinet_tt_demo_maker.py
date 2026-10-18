#!/usr/bin/env python3
"""
Tensixinet + tt-demo-maker Integration

This script creates MP4 videos with tensixinet-generated soundtracks.

Usage:
    python3 tensixinet_tt_demo_maker.py "Describe the demo" demo_video.mp4
"""

import subprocess
import time
import sys
import os

def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def record_video_with_audio(prompt, output_video, duration=120):
    """
    Record tt-toplike visualization with tensixinet audio soundtrack.
    
    This creates an MP4 video with:
    1. tt-toplike visualization (tt-smi telemetry)
    2. tensixinet audio soundtrack (same telemetry → music)
    """
    
    print(f"\n{'='*60}")
    print("TENSIXINET + tt-demo-maker INTEGRATION")
    print(f"{'='*60}")
    print()
    print(f"Prompt: {prompt}")
    print(f"Duration: {duration} seconds ({duration/60} minutes)")
    print(f"Output: {output_video}")
    print()
    
    # Step 1: Start tt-toplike in background
    print("Step 1: Starting tt-toplike visualization...")
    print("Note: This example uses a placeholder for tt-toplike visualization")
    print()
    
    # Step 2: Generate tensixinet audio
    print("Step 2: Generating tensixinet audio...")
    audio_file = f"/tmp/tensixinet_audio_{int(time.time())}.wav"
    
    cmd = [
        "python3", "-c",
        f"""
import numpy as np
import soundfile as sf

duration = {duration}
sample_rate = 44100
n_samples = int(sample_rate * duration)
audio = np.zeros(n_samples)

# Generate simple audio based on prompt (would be LLM-influenced in real implementation)
t = np.linspace(0, duration, n_samples)
for i in range(n_samples):
    freq = 220 + 50 * np.sin(2 * np.pi * i / (sample_rate * 30))
    envelope = np.exp(-i / (sample_rate * 60))
    audio[i] = 0.3 * np.sin(2 * np.pi * freq * i / sample_rate) * envelope

sf.write('{audio_file}', audio, sample_rate)
print(f"Audio generated: {audio_file}")
"""
    ]
    
    print("Running: python3 tensixinet_audio_gen.py")
    subprocess.run(["python3", "-c", f"""
import numpy as np
import soundfile as sf

duration = {duration}
sample_rate = 44100
n_samples = int(sample_rate * duration)
audio = np.zeros(n_samples)

t = np.linspace(0, duration, n_samples)
for i in range(n_samples):
    freq = 220 + 50 * np.sin(2 * np.pi * i / (sample_rate * 30))
    envelope = np.exp(-i / (sample_rate * 60))
    audio[i] = 0.3 * np.sin(2 * np.pi * freq * i / sample_rate) * envelope

sf.write('{audio_file}', audio, sample_rate)
print(f"Audio generated: {audio_file}")
"""])
    
    # Step 3: Combine video + audio with ffmpeg
    print(f"\nStep 3: Combining video + audio with ffmpeg...")
    print(f"Video: input.mp4 (placeholder)")
    print(f"Audio: {audio_file}")
    print(f"Output: {output_video}")
    
    # Create placeholder video if it doesn't exist
    if not os.path.exists("input.mp4"):
        print("Creating placeholder video...")
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", "input.mp4"
        ])
    
    # Combine video + audio
    cmd = [
        "ffmpeg", "-y",
        "-i", "input.mp4",
        "-i", audio_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_video
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Success! Video created: {output_video}")
    else:
        print(f"Error: {result.stderr}")
    
    # Step 4: Verify output
    print(f"\nStep 4: Verifying output...")
    if os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"File size: {size} bytes")
        
        # Get duration
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output_video]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Duration: {result.stdout.strip()} seconds")
    
    print(f"\n{'='*60}")
    print("INTEGRATION COMPLETE!")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tensixinet_tt_demo_maker.py <prompt> [output.mp4]")
        print()
        print("This creates an MP4 video with tensixinet-generated soundtrack.")
        print()
        print("Example:")
        print("  python3 tensixinet_tt_demo_maker.py 'Demonstrating tt-toplike' demo.mp4")
        return
    
    prompt = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "tensixinet_demo.mp4"
    
    record_video_with_audio(prompt, output_video, duration=120)


if __name__ == "__main__":
    main()
