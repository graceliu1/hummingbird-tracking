"""Identify bird visits to feeders in a video."""
import json
import math
import csv
import os
import datetime
import argparse

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def feeder_center(feeder):
    xmin, ymin, xmax, ymax = feeder["bbox"]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return (cx, cy)

def analyze_feeder_visits(
    frames,
    feeders,
    dist_threshold_feeder=15.0,
    dist_threshold_neighbor=25.0,
    gap_tolerance=5,
    min_visit_duration=30
):
    bird_at_feeder = {}

    for frame_data in frames:
        frame_id = frame_data["frame_id"]
        detections = [d for d in frame_data["detections"] if d["track_id"] is not None]

        for det in detections:
            t_id = det["track_id"]
            centroid = det["centroid"]

            for feeder in feeders:
                fx, fy = feeder_center(feeder)
                d = euclidean_distance(centroid, (fx, fy))

                if d <= dist_threshold_feeder:
                    if t_id not in bird_at_feeder:
                        bird_at_feeder[t_id] = {}
                    bird_at_feeder[t_id][frame_id] = feeder["label"]
                    break

    visits = []
    for t_id, frame_data in bird_at_feeder.items():
        frames_sorted = sorted(frame_data.items())
        if not frames_sorted:
            continue

        current_visit = {
            "track_id": t_id,
            "feeder_label": frames_sorted[0][1],
            "start_frame": frames_sorted[0][0],
            "end_frame": frames_sorted[0][0],
            "frames": [frames_sorted[0][0]]
        }

        for i in range(1, len(frames_sorted)):
            frame_id, feeder_label = frames_sorted[i]
            prev_frame_id = frames_sorted[i-1][0]

            if (feeder_label == current_visit["feeder_label"] and
                frame_id - prev_frame_id <= gap_tolerance + 1):
                current_visit["end_frame"] = frame_id
                current_visit["frames"].append(frame_id)
            else:
                visits.append(current_visit)
                current_visit = {
                    "track_id": t_id,
                    "feeder_label": feeder_label,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "frames": [frame_id]
                }

        visits.append(current_visit)

    visits.sort(key=lambda x: x["start_frame"])

    visits = [
        v for v in visits
        if (v["end_frame"] - v["start_frame"] + 1) >= min_visit_duration
    ]

    for visit in visits:
        start_frame = visit["start_frame"]
        t_id = visit["track_id"]

        start_frame_data = next((f for f in frames if f["frame_id"] == start_frame), None)
        birds_near_feeder = False

        if start_frame_data:
            for det in start_frame_data["detections"]:
                other_id = det["track_id"]
                if other_id is None or other_id == t_id:
                    continue
                for feeder in feeders:
                    fx, fy = feeder_center(feeder)
                    d = euclidean_distance(det["centroid"], (fx, fy))
                    if d <= dist_threshold_neighbor:
                        birds_near_feeder = True
                        break
                if birds_near_feeder:
                    break

        if birds_near_feeder:
            visit["validity"] = "invalid"
            visit["reason"] = "Another bird near a feeder at start frame"
        else:
            visit["validity"] = "valid"
            visit["reason"] = "No other birds near feeders at start frame"

    seen_tracks = set()
    for visit in visits:
        t_id = visit["track_id"]
        if t_id in seen_tracks:
            visit["validity"] = "invalid"
            visit["reason"] = f"Bird {t_id} already had a prior visit"
        else:
            seen_tracks.add(t_id)

    results = []
    for visit in visits:
        results.append({
            "start_frame": visit["start_frame"],
            "end_frame": visit["end_frame"],
            "track_id": visit["track_id"],
            "feeder_label": visit["feeder_label"],
            "validity": visit["validity"],
            "reason": visit["reason"]
        })

    return results

def frame_to_timestamp(frame_id, fps):
    return frame_id / fps

def format_time_hhmmss(seconds, start_dt):
    actual_time = start_dt + datetime.timedelta(seconds=seconds)
    return actual_time.strftime("%H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="Identify bird visits to feeders in a video.")
    parser.add_argument("--tracked", type=str, required=True, help="Path to tracked_bboxes.json")
    parser.add_argument("--feeders", type=str, required=True, help="Path to feeder_locations.json")
    parser.add_argument("--start_time", type=str, default="00:00:00", help="Video start time (format HH:MM:SS)")
    parser.add_argument("--fps", type=float, default=60.0, help="Frames per second of the video")

    args = parser.parse_args()

    with open(args.tracked, "r") as f:
        frames = json.load(f)
    with open(args.feeders, "r") as f:
        feeders = json.load(f)

    start_dt = datetime.datetime.strptime(args.start_time, "%H:%M:%S")

    visits = analyze_feeder_visits(
        frames,
        feeders,
        dist_threshold_feeder=15.0,
        dist_threshold_neighbor=100.0,
        gap_tolerance=10,
        min_visit_duration=10
    )

    for v in visits:
        v["start_time_seconds"] = frame_to_timestamp(v["start_frame"], args.fps)
        v["end_time_seconds"] = frame_to_timestamp(v["end_frame"], args.fps)

    visits.sort(key=lambda v: v["start_time_seconds"])

    output_path = os.path.join(os.path.dirname(os.path.abspath(args.tracked)), "visit_data.csv")

    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "start_time_seconds",
            "end_time_seconds",
            "start_frame",
            "end_frame",
            "track_id",
            "feeder_label",
            "validity",
            "reason"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for v in visits:
            writer.writerow({
                "start_time_seconds": round(v["start_time_seconds"], 2),
                "end_time_seconds": round(v["end_time_seconds"], 2),
                "start_frame": v["start_frame"],
                "end_frame": v["end_frame"],
                "track_id": v["track_id"],
                "feeder_label": v["feeder_label"],
                "validity": v["validity"],
                "reason": v["reason"]
            })

    print(f"Done! Visit CSV saved to {output_path}")

if __name__ == "__main__":
    main()
