import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_folder):
    """
    Extract all frames from a video file and save them as images.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder where frames will be saved
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Output folder: {output_folder}")
    print()
    
    frame_count = 0
    
    while True:
        # Read frame by frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame as image
        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # Print progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        frame_count += 1
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nExtraction completed!")
    print(f"Total frames extracted: {frame_count}")
    print(f"Frames saved to: {output_folder}")

def main():
    # Define paths
    video_path = "assets/GX011934.MP4"
    output_folder = "extracted_frames_left"
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    print(f"Starting frame extraction from {video_path}")
    print("=" * 50)
    
    # Extract frames
    extract_frames(video_path, output_folder)

if __name__ == "__main__":
    main()
