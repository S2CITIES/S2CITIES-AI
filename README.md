# S2CITIES-AI
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

Video-based Recognition of the "The Canadian Women's Foundation" Signal for Help

## Installation

1. Clone this repository with `git clone https://github.com/S2CITIES/S2CITIES-AI`
2. Install the dependencies with `pip install -r requirements.txt`

## Usage

### Dataset creation pipeline

1. Run the script [`dataset_creation_move_and_split.py`](./dataset_creation_move_and_split.py) to move, rename and split the videos.
2. Run the script [`dataset_creation_starter_csv.py`](./dataset_creation_starter_csv.py) to create the starter CSV file and facilitate labeling by someone else.
3. Run the script [`dataset_creation_perform_labeling.py`](./dataset_creation_perform_labeling.py) to actually perform the labeling.
4. Run the script [`dataset_creation_move_labeled.py`](./dataset_creation_move_labeled.py) to move the labeled videos into the respective class folders according to the CSV file.

### Analyse a dataset

To analyse a dataset use the following command

```bash
python analyse_dataset.py --dataset_path "/Users/teo/Library/CloudStorage/OneDrive-PolitecnicodiMilano/ASP/S2Cities/S2C - Machine Learning/Dataset/S2Cities_Dataset_Collection"
```

### mpkpts pipeline

To subsample the videos use the following command

```bash
python dataset_creation_subsample_videos.py --input "/Users/teo/My Drive (s2cities.project@gmail.com)/DRIVE S2CITIES/Artificial Intelligence/SFH_Dataset_S2CITIES/SFH_Dataset_S2CITIES_raw_extended_negatives" --output "data/5_videos_labeled_subsampled"
```

To extract the keypoints from the videos use the following command

```bash
python mpkpts_extract_keypoints.py --input "data/5_videos_labeled_subsampled" --output "data/6_features_extracted"
```

To extract the timeseries features from the keypoints use the following command

```bash
python mpkpts_extract_timeseries_features.py
```

To perfrom the train test split use the following command

```bash
python mpkpts_split_train_test.py
```

To perform the feature selection use the following command

```bash
python mpkpts_feature_selection.py
```

After running the feature selection, you have to choose the best features eyeballing the plot by inspecting and running the `mpkpts_visualize.ipynb` notebook.

To train the model use the following command

```bash
python mpkpts_train.py
```

To evaluate the model use the following command

```bash
python mpkpts_evaluate.py
```



## Installing TensorFlow + MediaPipe on Apple Silicon

**DEPRECATED**: This is probably not needed anymore, since the latest version of MediaPipe now supports macOS natively.

1. ~~Install `pip install mediapipe-silicon` which installs `protobuf==3.20.3`.~~
2. ~~Take the `builder.py` from the installed version and copy it somewhere else `~/.pyenv/versions/s2cities/lib/python3.10/site-packages/google/protobuf/internal`.~~
3. ~~Install `pip install tensorflow-macos tensorflow-metal` which installs `protobuf==3.19.6`. TensorFlow has priority, so keep this version, but overwrite the `builder.py` with the one you took earlier.~~

https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal

## Resources

### Mediapipe

- https://mediapipe-studio.webapps.google.com/demo/hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#image
- https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN

### General computer vision

- [Advanced Computer Vision with Python - Full Course](https://www.youtube.com/watch?v=01sAkU_NvOY)
- [CS231n Winter 2016: Lecture 14: Videos and Unsupervised Learning](https://www.youtube.com/watch?v=ekyBklxwQMU)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (too hard)

## Authors

- Teo Bucci ([@teobucci](https://github.com/teobucci))
- Dario Cavalli ([@Cavalli98](https://github.com/Cavalli98))
- Giuseppe Stracquadanio ([@pestrstr](https://github.com/pestrstr))
