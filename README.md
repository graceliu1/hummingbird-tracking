# It’s a Bird, Not a Plane: Automated Multi-Hummingbird Tracking in Videos

Grace Liu (COS IW 08: CV for Social Good, Spring 2025)

Advisors: Olga Russakovsky, Sarah Solie

Supported by the Stoddard Lab at Princeton.

## Procedure
1. Clone this repository. Make sure to install requirements with ```pip install -r requirements.txt```.
2. Create folder titled ```[VIDEO_NAME]``` inside ```/datasets```.
3. Upload ```[VIDEO_NAME].mp4``` under said folder.
4. ```chmod +x run_pipeline.sh```
5. ```./run_pipeline.sh```

All output is under ```/datasets/[VIDEO_NAME]```.

## Outputs
* ```mot_[VIDEO_NAME].mp4```
* ```visit_data.csv```
