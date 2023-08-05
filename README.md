<!-- omit from toc -->
# S2CITIES-AI
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

Video-based Recognition of the "The Canadian Women's Foundation" Signal for Help

- [Installation](#installation)
- [Usage](#usage)
  - [Dataset creation pipeline](#dataset-creation-pipeline)
  - [Analyse a dataset](#analyse-a-dataset)
  - [mpkpts pipeline](#mpkpts-pipeline)
- [Resources](#resources)
  - [Mediapipe](#mediapipe)
  - [General computer vision](#general-computer-vision)
- [Authors](#authors)

## Installation

1. Clone this repository with `git clone https://github.com/S2CITIES/S2CITIES-AI`
2. Install the dependencies with `pip install -r requirements.txt`

## Usage

### Dataset creation pipeline

1. Run the following to move, rename and split the videos, making sure to set the `starting_idx` parameter to the starting index to use (i.e. the index of the last video + 1).
```bash
python dataset_creation_move_and_split.py \
--starting_idx 123 \
--path_videos_arrived "data/0_videos_arrived" \
--path_videos_raw "data/1_videos_raw" \
--path_videos_raw_processed "data/2_videos_raw_processed" \
--path_videos_splitted "data/3_videos_splitted"
```
2. Run the following to create the starter CSV file and facilitate labeling by someone else.
```bash
python dataset_creation_starter_csv.py \
--folder "data/3_videos_splitted" \
--csv_filename "data/labels.csv"
```
3. Run the following to actually perform the labeling.
```bash
python dataset_creation_perform_labeling.py \
--folder "data/3_videos_splitted" \
--csv_filename "data/labels.csv"
```
4. Run the following to move the labeled videos into the respective class folders according to the CSV file.
```bash
python dataset_creation_move_labeled.py \
--source_folder "data/3_videos_splitted" \
--destination_folder "data/4_videos_labeled" \
--csv_filename "data/labels.csv"
```

Now `data/4_videos_labeled` should contain the labeled videos, under the `1` and `0` folders.

### Analyse a dataset

To analyse a dataset use the following command

```bash
python analyse_dataset.py --dataset_path "/Users/teo/Library/CloudStorage/OneDrive-PolitecnicodiMilano/ASP/S2Cities/S2C - Machine Learning/Dataset/S2Cities_Dataset_Collection"
```

### mpkpts pipeline

1. To subsample the videos use the following command

```bash
python dataset_creation_subsample_videos.py \
--input "data/4_videos_labeled" \
--output "data/5_videos_labeled_subsampled"
```

Remember that if the dataset has been modified or moved, for example on a mounted drive, you can easily replace the input folder:

```bash
python dataset_creation_subsample_videos.py \
--input "/Users/teo/My Drive (s2cities.project@gmail.com)/DRIVE S2CITIES/Artificial Intelligence/SFH_Dataset_S2CITIES/SFH_Dataset_S2CITIES_raw_extended_negatives" \
--output "data/5_videos_labeled_subsampled"
```

2. To extract the keypoints from the videos use the following command

```bash
python mpkpts_extract_keypoints.py \
--input "data/5_videos_labeled_subsampled" \
--output "data/6_features_extracted"
```

3. To extract the timeseries features from the keypoints use the following command

```bash
python mpkpts_extract_timeseries_features.py \
--input "data/6_features_extracted" \
--output "data/7_timeseries_features_extracted"
```

4. To perfrom the train test split use the following command

```bash
python mpkpts_split_train_test.py --folder "data/7_timeseries_features_extracted"
```

optionally, you can specify the `--test_size` parameter to change the size of the test set (default is 0.2).

5. To perform the feature selection use the following command

```bash
python mpkpts_feature_selection.py --folder "data/7_timeseries_features_extracted"
```

optionally, you can specify the `--n_jobs` parameter to change the number of jobs to run in parallel (default is 1).

After running the feature selection, you have to choose the best features eyeballing the plot by inspecting and running the `mpkpts_visualize.ipynb` notebook.

6. To train the model use the following command

```bash
python mpkpts_train.py --folder "data/7_timeseries_features_extracted"
```

7. To evaluate the model use the following command

```bash
python mpkpts_evaluate.py --folder "data/7_timeseries_features_extracted"
```

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
