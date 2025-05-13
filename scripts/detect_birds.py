"""Object detection script for bird tracking using background subtraction."""
import cv2
import numpy as np
import os
import glob
import json
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

MIN_CONTOUR_AREA = 85
SEGMENT_SIZE = 2000

def compute_mean_background(frame_paths):
    accum = None
    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp).astype(np.float32)
        if accum is None:
            accum = frame
        else:
            alpha = 1.0 / (i + 1)
            accum = (1 - alpha) * accum + alpha * frame
    return accum.astype(np.uint8)

def compute_segmented_backgrounds(frame_paths, segment_size):
    backgrounds, indices = [], []
    for i in range(0, len(frame_paths), segment_size):
        segment = frame_paths[i:i + segment_size]
        backgrounds.append(compute_mean_background(segment))
        indices.append((i, i + len(segment)))
    return backgrounds, indices

def get_background_for_frame(frame_id, backgrounds, indices):
    for (start, end), bg in zip(indices, backgrounds):
        if start <= frame_id < end:
            return bg
    return backgrounds[-1]

def create_timer_mask(shape, region):
    mask = np.ones(shape[:2], dtype=np.uint8) * 255
    x, y, w, h = region
    mask[y:y+h, x:x+w] = 0
    return mask

def preprocess_frame(frame, background, timer_mask):
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if timer_mask is not None:
        gray = cv2.bitwise_and(gray, timer_mask)
    return cv2.GaussianBlur(gray, (3, 3), 0)

def threshold_frame(blurred):
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

def is_overlapping(box, region):
    x1, y1, x2, y2 = box
    rx, ry, rw, rh = region
    return not (x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh)

def detect_objects(frame, mask, frame_id, annotate, out_dir, timer_region=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if timer_region and is_overlapping((x, y, x + w, y + h), timer_region):
            continue
        detections.append({
            "bbox": [x, y, x + w, y + h],
            "centroid": [x + w // 2, y + h // 2],
            "track_id": None
        })
        if annotate:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if annotate and frame_id is not None:
        if timer_region:
            tx, ty, tw, th = timer_region
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"detected_birds_{frame_id:04d}.png"), frame)

    return {"frame_id": frame_id, "detections": detections}

def process_frame(path, backgrounds, indices, annotate, out_dir, timer_mask=None, timer_region=None):
    frame = cv2.imread(path)
    match = re.search(r"(\d+)", os.path.basename(path))
    frame_id = int(match.group(1)) if match else None
    background = get_background_for_frame(frame_id, backgrounds, indices)
    blurred = preprocess_frame(frame, background, timer_mask)
    mask = threshold_frame(blurred)
    return detect_objects(frame.copy(), mask, frame_id, annotate, out_dir, timer_region)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect birds using background subtraction.")
    parser.add_argument("--input", required=True, help="Input frame folder.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--save_marked_frames", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--mask_timer", action="store_true")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    annotate = args.save_marked_frames

    os.makedirs(output_dir, exist_ok=True)
    annotated_dir = os.path.join(output_dir, "bird_detections")
    if annotate:
        os.makedirs(annotated_dir, exist_ok=True)

    all_frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not all_frames:
        raise RuntimeError("No frames found.")

    print(f"Processing {len(all_frames)} frames...")

    bg_array_path = os.path.join(output_dir, "backgrounds.npz")
    bbox_json_path = os.path.join(output_dir, "bboxes.json")

    if os.path.exists(bg_array_path):
        data = np.load(bg_array_path)
        backgrounds = [data[k] for k in sorted(data.files)]
        segment_count = len(backgrounds)
        frames_per_segment = max(1, len(all_frames) // segment_count)
        indices = [
            (i * frames_per_segment,
             len(all_frames) if i == segment_count - 1 else (i + 1) * frames_per_segment)
            for i in range(segment_count)
        ]
        print(f"Loaded {segment_count} background segments.")
    else:
        segment_size = min(SEGMENT_SIZE, len(all_frames))
        print("Computing mean backgrounds...")
        backgrounds, indices = compute_segmented_backgrounds(all_frames, segment_size)
        np.savez_compressed(bg_array_path, **{f"bg_{i}": bg for i, bg in enumerate(backgrounds)})
        print(f"Saved backgrounds to {bg_array_path}")

    timer_region = None
    timer_mask = None
    if args.mask_timer:
        h, w = backgrounds[0].shape[:2]
        timer_region = [int(w * 0.85), int(h * 0.95), int(w * 0.15), int(h * 0.05)]
        timer_mask = create_timer_mask((h, w, 3), timer_region)
        cv2.imwrite(os.path.join(output_dir, "timer_mask.png"), timer_mask)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(
                lambda p: process_frame(p, backgrounds, indices, annotate, annotated_dir, timer_mask, timer_region),
                all_frames
            ),
            total=len(all_frames)
        ))

    with open(bbox_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processing complete. Saved results to {bbox_json_path}")