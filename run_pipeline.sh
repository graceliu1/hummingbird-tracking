#!/bin/bash
set -e
trap "echo 'Interrupted. Exiting...'; exit 1" SIGINT SIGTERM

start_time=$(date +%s)
echo "=== Started at $(date '+%Y-%m-%d %H:%M:%S') ==="

DATASET="virginia"
START_FRAME=0
NUM_FRAMES=2000
FPS=60
START_TIMESTAMP="08:22:40"
TRACKED_OUTPUT_VIDEO="mot_virginia.mp4"

dataset="datasets/$DATASET"
video="$dataset/${DATASET}.MP4"
frames="$dataset/frames"
boxes_out="$dataset/bboxes.json"
background_out="$dataset/mean_background.png"
tracked_out="$dataset/tracked_bboxes.json"
tracked_frames_dir="$dataset/tracked_frames"
output_video="$dataset/$TRACKED_OUTPUT_VIDEO"
feeders_file="$dataset/feeder_locations.json"
visit_csv="$dataset/visit_data.csv"

if [ ! -f "$video" ]; then
    echo "Video file not found: $video"
    exit 1
fi

echo "Processing $DATASET.MP4"

echo "Extracting frames..."
rm -rf "$frames"
python scripts/extract_frames.py \
    --input "$video" \
    --output "$frames" \
    --start "$START_FRAME" \
    --num-frames "$NUM_FRAMES"

echo "Running bird detection..."
python scripts/detect_birds.py \
    --input "$frames" \
    --output "$dataset" \
    --mask_timer

echo "Running tracking..."
python scripts/tracker.py \
    --input "$boxes_out" \
    --output "$dataset" \
    --visualize \
    --frames "$frames"

echo "Creating video..."
python scripts/frames_to_video.py \
    --input "$tracked_frames_dir" \
    --output "$dataset" \
    --fps "$FPS" \
    --video_name "$TRACKED_OUTPUT_VIDEO"

echo "Launching feeder selection..."
python scripts/draw_feeders.py --video "$video"

echo "Generating visit CSV..."
python scripts/identify_visits.py \
    --tracked "$tracked_out" \
    --feeders "$feeders_file" \
    --start_time "$START_TIMESTAMP" \
    --fps "$FPS"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "=== Finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Total runtime: $((duration / 60)) min $((duration % 60)) sec"
echo "Pipeline completed."
