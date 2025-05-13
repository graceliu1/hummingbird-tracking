import os
import json
import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import argparse
from tqdm import tqdm

IOU_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 50
MAX_AGE = 10
MIN_HIT_STREAK = 15
REMAPPING_THRESHOLD = 10

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_centroid_distance(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return np.hypot(cxA - cxB, cyA - cyB)

def bbox_to_state(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    w, h = x2 - x1, y2 - y1
    return np.array([cx, cy, w, h])

def state_to_bbox(state):
    cx, cy, w, h = state[:4]
    return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

def create_kalman_filter(initial_state):
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 1.0
    kf.F = np.eye(8)
    for i in range(4):
        kf.F[i, i+4] = dt
    kf.H = np.eye(4, 8)
    kf.R *= 10.
    kf.P *= 1000.
    kf.Q *= 1.0
    kf.x[:4] = initial_state.reshape((4,1))
    return kf

def remap_track_ids(data, threshold=REMAPPING_THRESHOLD):
    usage_count = defaultdict(int)
    for frame in data:
        for det in frame["detections"]:
            tid = det.get("track_id")
            if tid is not None:
                usage_count[tid] += 1
    valid_ids = sorted([tid for tid, count in usage_count.items() if count >= threshold])
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_ids)}

    for frame in data:
        new_detections = []
        for det in frame["detections"]:
            tid = det.get("track_id")
            if tid is not None and tid in id_mapping:
                det["track_id"] = id_mapping[tid]
                new_detections.append(det)
        frame["detections"] = new_detections

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input bboxes.json")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--frames", help="Path to input frames directory")
    args = parser.parse_args()

    with open(args.input) as f:
        all_detections = json.load(f)

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)
    tracked_json_path = os.path.join(output_root, "tracked_bboxes.json")
    tracked_frames_dir = os.path.join(output_root, "tracked_frames") if args.visualize else None
    if args.visualize:
        os.makedirs(tracked_frames_dir, exist_ok=True)

    next_track_id = 0
    active_tracks = {}
    tentative_tracks = {}

    for frame_data in all_detections:
        detections = frame_data['detections']
        updated_tracks = {}
        matched_track_ids = set()

        predicted_positions = {}
        track_ids = list(active_tracks.keys()) + list(tentative_tracks.keys())
        for tid in track_ids:
            track = active_tracks.get(tid) or tentative_tracks.get(tid)
            track['kf'].predict()
            predicted_positions[tid] = state_to_bbox(track['kf'].x.flatten())

        detection_bboxes = [det['bbox'] for det in detections]
        cost_matrix = []
        for det_bbox in detection_bboxes:
            row = []
            for tid in track_ids:
                pred_bbox = predicted_positions[tid]
                iou = compute_iou(det_bbox, pred_bbox)
                dist = compute_centroid_distance(det_bbox, pred_bbox)
                cost = -iou if iou >= IOU_THRESHOLD else (-0.001 if dist < DISTANCE_THRESHOLD else 1e6)
                row.append(cost)
            cost_matrix.append(row)

        assigned = set()
        if cost_matrix:
            row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r][c] < 1e5:
                    det = detections[r]
                    tid = track_ids[c]
                    track = active_tracks.get(tid) or tentative_tracks.get(tid)
                    track['kf'].update(bbox_to_state(det['bbox']).reshape((4,1)))
                    track['hit_streak'] += 1
                    track['frames_since_seen'] = 0
                    track['total_visible_count'] += 1
                    if tid in active_tracks:
                        updated_tracks[tid] = track
                        if track['hit_streak'] >= MIN_HIT_STREAK:
                            det['track_id'] = tid
                    else:
                        if track['hit_streak'] >= MIN_HIT_STREAK:
                            active_tracks[tid] = track
                            updated_tracks[tid] = track
                            det['track_id'] = tid
                            del tentative_tracks[tid]
                        else:
                            tentative_tracks[tid] = track
                    matched_track_ids.add(tid)
                    assigned.add(r)

        for i, det in enumerate(detections):
            if i not in assigned:
                kf = create_kalman_filter(bbox_to_state(det['bbox']))
                tentative_tracks[next_track_id] = {
                    'kf': kf,
                    'frames_since_seen': 0,
                    'hit_streak': 1,
                    'total_visible_count': 1
                }
                next_track_id += 1

        for tid in list(active_tracks.keys()):
            if tid not in matched_track_ids:
                active_tracks[tid]['frames_since_seen'] += 1
                if active_tracks[tid]['frames_since_seen'] <= MAX_AGE:
                    updated_tracks[tid] = active_tracks[tid]

        for tid in list(tentative_tracks.keys()):
            if tid not in matched_track_ids:
                tentative_tracks[tid]['frames_since_seen'] += 1
                if tentative_tracks[tid]['frames_since_seen'] > MAX_AGE:
                    del tentative_tracks[tid]

        active_tracks = updated_tracks

    remapped_data = remap_track_ids(all_detections)
    with open(tracked_json_path, 'w') as f:
        json.dump(remapped_data, f, indent=2)

    if args.visualize:
        colors = [(57,255,20),(255,20,147),(0,255,255),(255,255,0),(255,0,255),(0,255,127),(0,191,255),(255,105,180),(173,255,47),(127,255,212)]
        for frame_data in tqdm(remapped_data):
            frame_id = frame_data['frame_id']
            frame_path = os.path.join(args.frames, f"frame_{frame_id:04d}.png")
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            for det in frame_data['detections']:
                if 'track_id' not in det:
                    continue
                x1, y1, x2, y2 = det['bbox']
                tid = det['track_id']
                color = colors[tid % len(colors)]
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out_path = os.path.join(tracked_frames_dir, f"tracked_{frame_id:04d}.png")
            cv2.imwrite(out_path, frame)

if __name__ == '__main__':
    main()
