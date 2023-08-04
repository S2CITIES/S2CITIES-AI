import argparse
from src import constants
from src.dataset_creation.videosubsampler import VideoSubsampler

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Subsample videos.')
    parser.add_argument('--input', type=str, required=True, help='Path to input directory.')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory.')
    args = parser.parse_args()

    subsampler = VideoSubsampler(
        target_fps=constants.SUBSAMPLE_FPS,
        video_extensions=constants.VIDEO_EXTENSIONS,
        path_input=args.input,
        path_output=args.output,
        )

    subsampler.subsample_videos()
