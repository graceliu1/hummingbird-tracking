"""Extract frames from input video."""
import os
import cv2
import argparse
from tqdm import tqdm
import concurrent.futures

def extract_frame(frame_data):
    frame_id, frame, output_dir = frame_data
    filename = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
    cv2.imwrite(filename, frame)
    return frame_id

def extract_frames(input_path, output_dir, start=0, num_frames=2000, num_threads=None):
    if num_threads is None:
        num_threads = os.cpu_count()

    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_path}")
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = start + num_frames
    
    if start < 0 or start >= total_video_frames:
        raise ValueError(f"start ({start}) must be between 0 and {total_video_frames - 1}")
    
    if end > total_video_frames:
        raise ValueError(f"start + num_frames ({end}) exceeds total frames ({total_video_frames})")
    
    print(f"Extracting {num_frames} frames (from {start} to {end - 1}) from '{input_path}' into '{output_dir}'...")
    print(f"Using {num_threads} threads")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
   
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    futures = []
    
    frame_id = start
    read_count = 0
    
    pbar = tqdm(total=num_frames, desc="Extracting frames", unit="frame")
    
    batch_size = min(100, num_frames)  # Process in batches of 100 frames
    while frame_id < end:
        current_batch = []
        # Read a batch of frames
        for _ in range(batch_size):
            if frame_id >= end:
                break
                
            ret, frame = cap.read()
            if not ret:
                print(f"Stopped early: could not read frame at index {frame_id}.")
                break
                
            current_batch.append((frame_id, frame.copy(), output_dir))
            frame_id += 1
            read_count += 1
            
        # Process current batch in parallel
        batch_futures = [executor.submit(extract_frame, data) for data in current_batch]
        futures.extend(batch_futures)
        
        for _ in range(len(current_batch)):
            pbar.update(1)
    
    executor.shutdown(wait=True)
    
    cap.release()
    pbar.close()
    print(f"Done. Extracted {read_count} frames to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to input video file. Required.")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to output directory. Required.")
    parser.add_argument("--start", type=int, default=0,
                      help="Starting frame index (inclusive). Default: 0")
    parser.add_argument("--num-frames", type=int, default=2000,
                      help="Number of frames to extract. Default: 2000")
    parser.add_argument("--num-threads", type=int, default=None,
                      help="Number of threads to use. Default: number of CPU cores")
    args = parser.parse_args()
    
    extract_frames(args.input, args.output, args.start, args.num_frames, args.num_threads)