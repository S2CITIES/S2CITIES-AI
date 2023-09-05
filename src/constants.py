data_folder = "data/"

VIDEOS_ARRIVED = data_folder + "0_videos_arrived"
VIDEOS_RAW = data_folder + "1_videos_raw"
VIDEOS_RAW_PROCESSED = data_folder + "2_videos_raw_processed"
VIDEOS_SPLITTED = data_folder + "3_videos_splitted"
VIDEOS_LABELED = data_folder + "4_videos_labeled"
VIDEOS_LABELED_SUBSAMPLED = data_folder + "5_videos_labeled_subsampled"
FEATURES_EXTRACTED = data_folder + "6_features_extracted"
TIMESERIES_FEATURES_EXTRACTED = data_folder + "7_timeseries_features_extracted"
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
SUBSAMPLE_FPS = 12

SUBCLIP_DURATION = 2.5
SHIFT_DURATION = 1
SEED = 42
MODEL_NAME = "mpkpts"
LABELS = ["SFH_No", "SFH_Yes"]

LABELS_DICT = {
    0: "SFH_No",
    1: "SFH_Yes",
}

REPORTS_FOLDER = "report/"
REPORTS_FOLDER_MPKPTS = REPORTS_FOLDER + "mpkpts/"
