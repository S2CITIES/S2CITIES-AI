# S2CITIES-AI
Video-based Recognition of the "The Canadian Women's Foundation" Signal for Help

## Installing TensorFlow + MediaPipe on Apple Silicon

1. Install `pip install mediapipe-silicon` which installs `protobuf==3.20.3`.
2. Take the `builder.py` from the installed version and copy it somewhere else `~/.pyenv/versions/s2cities/lib/python3.10/site-packages/google/protobuf/internal`.
3. Install `pip install tensorflow-macos tensorflow-metal` which installs `protobuf==3.19.6`. TensorFlow has priority, so keep this version, but overwrite the `builder.py` with the one you took earlier.

https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal

## Resources

- [Advanced Computer Vision with Python - Full Course](https://www.youtube.com/watch?v=01sAkU_NvOY)
- [CS231n Winter 2016: Lecture 14: Videos and Unsupervised Learning](https://www.youtube.com/watch?v=ekyBklxwQMU)
- [Mediapipe](https://google.github.io/mediapipe/)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470
- https://arxiv.org/pdf/1704.04861.pdf
- https://paperswithcode.com/search?q_meta=&q_type=&q=resnet-10
- https://pure.port.ac.uk/ws/files/27762140/Dynamic_hand_gesture.pdf
- https://github.com/russointroitoa/Machine-Learning-Collection
- https://cs50.harvard.edu/x/2022/

## Usage

1. Follow the istructions to have the [`4_videos_labeled`](./src/dataset_creation/4_videos_labeled/) folder with the labeled videos
2. Run the script [`subsample_videos.py`](./src/subsample_videos.py) to subsample the videos to a predefined FPS
3. Run the script [`extract_features.py`](./src/extract_features.py) to extract the keypoints using MediaPipe
4. Run the script [`timeseries_feature_extraction.py`](./src/timeseries_feature_extraction.py) to extract features from the timeseries of keypoints using, momentarily, [`tsfresh`](https://tsfresh.readthedocs.io/)
5. Run the [`train_model_nn.py`](./src/train_model_nn.py) or the [`train_model_stats.py`](./src/train_model_stats.py) to train the model for classification.
