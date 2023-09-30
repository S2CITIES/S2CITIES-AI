<!-- omit from toc -->
# S2CITIES: Toward Smart and Safe Cities: Exploit- ing Surveillance Videos for Real-time Detection of “Signal for Help” ✋✊
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

Welcome the S2CITIES Artificial Intelligence project repository! 🚀

This project aims at developing a real-time AI system for the detection of the "Signal for Help" gesture in surveillance videos.

Give us a star to show your support for the project ⭐

Alta Scuola Politecnica (more [here](https://www.asp-poli.it/)) is the **joint honors program** of Italy's best technical universities, Politecnico di Milano ([18th world-wide, QS Rankings](https://www.topuniversities.com/university-rankings/university-subject-rankings/2023/engineering-technology?&page=1)) and Politecnico di Torino ([45th world-wide, QS Rankings](https://www.topuniversities.com/university-rankings/university-subject-rankings/2023/engineering-technology?&page=1)). 
Each year, 90 students from Politecnico di Milano and 60 from Politecnico di Torino are selected from a highly competitive pool and those who succeed receive free tuition for their MSc in exchange for ~1.5 years working as **student consultants** with a partner company for an industrial project.

This project was carried out alogside a research group at Politecnico di Torino (Prof. [Sarah Azimi](https://www.linkedin.com/in/sarah-azimi-539a3a7a/) and Prof. [Vincenzo Randazzo](https://www.linkedin.com/in/vincenzo-randazzo-19963189/)), backed up by [Getapper](https://www.getapper.com/) software company.


- [Installation](#installation)
- [Usage: Dataset creation pipeline 💡](#usage-dataset-creation-pipeline-)
  - [Analyse a dataset](#analyse-a-dataset)
- [Usage: MediaPipe + Time-Series Feature Extraction 💡](#usage-mediapipe--time-series-feature-extraction-)
  - [mpkpts pipeline](#mpkpts-pipeline)
  - [Real-time testing](#real-time-testing)
  - [Analyse the timeseries features](#analyse-the-timeseries-features)
- [Usage: Fine-tuning a 3D-CNN torch model 💡](#usage-fine-tuning-a-3d-cnn-torch-model-)
  - [Pre-processing data](#pre-processing-data)
  - [Generating annotations](#generating-annotations)
  - [Fine-tuning on custom dataset](#fine-tuning-on-custom-dataset)
- [Resources 📚](#resources-)
- [Authors 🖋](#authors-)

## Installation

1. Clone this repository with `git clone https://github.com/S2CITIES/S2CITIES-AI`
2. Install the dependencies with `pip install -r requirements.txt`

## Usage: Dataset creation pipeline 💡

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
python analyse_dataset.py \
--dataset_path "/Users/teo/My Drive (s2cities.project@gmail.com)/DRIVE S2CITIES/Artificial Intelligence/SFH_Dataset_S2CITIES/SFH_Dataset_S2CITIES_raw_extended_negatives" \
--save_dir "report/dataset_analysis"
```

## Usage: MediaPipe + Time-Series Feature Extraction 💡

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
python mpkpts_split_train_test.py \
--folder "data/7_timeseries_features_extracted"
```

optionally, you can specify the `--test_size` parameter to change the size of the test set (default is 0.2).

5. To perform the feature selection use the following command

```bash
python mpkpts_feature_selection.py \
--folder "data/7_timeseries_features_extracted"
```

optionally, you can specify the `--n_jobs` parameter to change the number of jobs to run in parallel (default is 1).

After running the feature selection, you have to choose the best features eyeballing the plot by inspecting and running the `mpkpts_visualize.ipynb` notebook.

6. To train the model use the following command

```bash
python mpkpts_train.py \
--folder "data/7_timeseries_features_extracted"
```

7. To evaluate the model use the following command

```bash
python mpkpts_evaluate.py \
--folder "data/7_timeseries_features_extracted"
```

8. To get the prediction time use the following command

```bash
python mpkpts_get_prediction_time.py \
--folder "data/7_timeseries_features_extracted" \
--training_results "data/7_timeseries_features_extracted/training_results.pkl"
```

### Real-time testing

To test the model in real-time, run the following command

```bash
python realtime_multithread.py \
--training_results "data/7_timeseries_features_extracted/training_results.pkl" \
--tsfresh_parameters "data/7_timeseries_features_extracted/kind_to_fc_parameters.pkl" \
--scaler "data/7_timeseries_features_extracted/scaler.pkl" \
--final_features "data/7_timeseries_features_extracted/final_features.pkl" \
--model_choice RF \
--threshold 0.5
```

### Analyse the timeseries features

To analyse the timeseries features use the following command

```bash
python analyse_timeseries.py \
--folder data/6_features_extracted \
--pos_index 1 \
--neg_index 1 \
--save_dir "report/timeseries_analysis"
```

## Usage: Fine-tuning a 3D-CNN torch model 💡

### Pre-processing data

A script is provided to pre-process train/test and validation data. Pre-processing helps in drastically speeding up the training process. Indeed, pre-processed videos will be saved in a memory location and retrieved while building training batches, so that the same pre-processing pipeline can be avoided while building batches accross all the training epochs.

Here, pre-processing means cropping videos to obtain videos with 1:1 aspect ratio and with selected target frame sizes.

To start the pre-processing script, run:

```bash
python data/SFHDataset/video_conversion_script.py \ 
--source_path "dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_raw" \
--target_width 112 \
--target_height 112
```

Depending on the size of your dataset stored in location `--source_path`, this may require a while.

### Generating annotations

After pre-processing your data, you can generate annotations files by simply running:

```bash
python data/SFHDataset/gen_annotations.py \
--data_path "dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_raw_ratio1_112x112"
```

This script will update the file `info.json` and write (**over-write**) the files `(train|test|val)_annotations.txt` in `data/SFHDataset`. It will also print some statistics regarding label (positive/negative) distributions on the train/test/val sets.

Statistics will be saved on file `info.json` for future reference. This file will also contain other useful information, such as the train set *mean* and *standard deviation*. 

### Fine-tuning on custom dataset

Now you have all the ingredients that you need to start training/fine-tuning a 3D-CNN model. 
To start your fine-tuning procedure, run:

```bash
python train_SFH_3dcnn.py \
--exp "mn2-sgd-nesterov-no-decay" \
--optimizer "SGD" \
--model "mobilenetv2" \
--data_path "dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_raw_ratio1_112x112" \ 
--early_stop_patience 20 \
--epochs 1000 \
--lr 1e-3 \
--sample_size 112 \
--train_crop corner \
--lr_patience 40
```

The command above will start to fine-tune a MobileNet-v2 model, pre-trained on the Jester dataset. \
Available models to fine-tune are (up-to-now): *MobileNet*, *MobileNet-v2*, *SqueezeNet*. \
You can specify the path to your pre-trained weights with `--pretrained_path`. \
You also may want to have a look at the script `train_args.py` to know all the possible parameters that you can specify.
Alternatively, you can simply run a `python train_SFH_3dcnn.py -h|--help`.

## Resources 📚

- https://mediapipe-studio.webapps.google.com/demo/hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#image
- https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN

## Authors 🖋

- Teo Bucci ([@teobucci](https://github.com/teobucci))
- Dario Cavalli ([@Cavalli98](https://github.com/Cavalli98))
- Giuseppe Stracquadanio ([@pestrstr](https://github.com/pestrstr))
