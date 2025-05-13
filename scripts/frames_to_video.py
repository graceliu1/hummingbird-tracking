"""Combine image frames into a video file with timestamps and frame numbers."""
import os
import cv2
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image frames to an MP4 video.")
    parser.add_argument("--input", type=str, required=True, help="Path to folder containing input frames.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--video_name", type=str, default=None, help="Output video filename (must end in .mp4).")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second. Default: 60.")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    fps = args.fps

    folder_name = os.path.basename(os.path.normpath(input_dir))
    video_filename = args.video_name or f"{folder_name}.mp4"

    if not video_filename.endswith(".mp4"):
        raise ValueError("Only .mp4 output is supported. Please provide --video_name ending in '.mp4'.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, video_filename)

    frame_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.png'))
    ])

    if not frame_files:
        raise ValueError(f"No image files found in {input_dir}")

    first_image_path = os.path.join(input_dir, frame_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"Failed to read first image: {frame_files[0]}")
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter. Check codec support and output path: {output_path}")

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.8
    font_thickness = 2
    padding = 80

    for i, filename in enumerate(tqdm(frame_files, desc="Writing video frames")):
        img_path = os.path.join(input_dir, filename)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping unreadable file {filename}")
            continue

        frame_number = i + 1
        timestamp_seconds = frame_number / fps

        label_frame = f"Frame {frame_number}"
        label_time = f"{timestamp_seconds:.2f} s"

        (frame_text_width, frame_text_height), _ = cv2.getTextSize(label_frame, font, font_scale, font_thickness)
        (time_text_width, time_text_height), _ = cv2.getTextSize(label_time, font, font_scale, font_thickness)

        max_text_width = max(frame_text_width, time_text_width)
        x = width - max_text_width - padding
        y_frame = frame_text_height + padding
        y_time = y_frame + frame_text_height + 10

        box_padding = 6
        box_top = y_frame - frame_text_height - box_padding
        box_bottom = y_time + box_padding

        cv2.rectangle(frame,
                      (x - box_padding, box_top),
                      (x + max_text_width + box_padding, box_bottom),
                      (0, 0, 0), -1)

        cv2.putText(frame, label_frame, (x, y_frame), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, label_time, (x, y_time), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved: {output_path}")

