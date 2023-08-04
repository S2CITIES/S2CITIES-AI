import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_video_grayscale(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize empty histogram bins
    bins = np.zeros((256,), dtype=int)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate histogram of the grayscale frame
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        bins = np.add(bins, hist.flatten().astype(int))

    cap.release()

    return bins

def analyze_video_rgb(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize empty histogram bins for each channel
    bins_r = np.zeros((256,), dtype=int)
    bins_g = np.zeros((256,), dtype=int)
    bins_b = np.zeros((256,), dtype=int)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate histogram of each channel
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])

        bins_r = np.add(bins_r, hist_r.flatten().astype(int))
        bins_g = np.add(bins_g, hist_g.flatten().astype(int))
        bins_b = np.add(bins_b, hist_b.flatten().astype(int))

    cap.release()

    return bins_r, bins_g, bins_b


def compute_average_distribution_grayscale(video_folder):
    videos = os.listdir(video_folder)
    total_videos = len(videos)
    average_distribution = np.zeros((256,), dtype=float)

    for video in videos:
        video_path = os.path.join(video_folder, video)
        pixel_distribution = analyze_video_grayscale(video_path)
        average_distribution = np.add(average_distribution, pixel_distribution)

    average_distribution /= total_videos

    return average_distribution

def compute_average_distribution_rgb(video_folder):
    videos = os.listdir(video_folder)
    total_videos = len(videos)
    average_distribution_r = np.zeros((256,), dtype=float)
    average_distribution_g = np.zeros((256,), dtype=float)
    average_distribution_b = np.zeros((256,), dtype=float)

    for video in videos:
        video_path = os.path.join(video_folder, video)
        bins_r, bins_g, bins_b = analyze_video_rgb(video_path)
        average_distribution_r = np.add(average_distribution_r, bins_r)
        average_distribution_g = np.add(average_distribution_g, bins_g)
        average_distribution_b = np.add(average_distribution_b, bins_b)

    average_distribution_r /= total_videos
    average_distribution_g /= total_videos
    average_distribution_b /= total_videos

    return average_distribution_r, average_distribution_g, average_distribution_b


if __name__ == '__main__':

    domains = ['grayscale', 'rgb']

    selected_domain = domains[1]

    if selected_domain == 'grayscale':
        # Provide the paths to the folders containing videos for each class
        negatives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/0'
        positives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/1'

        # Compute the average pixel distribution for each class
        average_distribution_negatives = compute_average_distribution_grayscale(negatives_folder)
        average_distribution_positives = compute_average_distribution_grayscale(positives_folder)

        # Plot the average pixel distributions of both classes on the same plot
        plt.plot(average_distribution_negatives, label='Negatives')
        plt.plot(average_distribution_positives, label='Positives')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Avg. Pixel Distribution - Original Dataset')
        plt.legend()
        plt.savefig('data/SFHDataset/analysis/frequency_analysis_grayscale_new_dataset.pdf', format='pdf')
        plt.close()

        # negatives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/0'
        # positives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/1'

        # # Compute the average pixel distribution for each class
        # average_distribution_negatives = compute_average_distribution_grayscale(negatives_folder)
        # average_distribution_positives = compute_average_distribution_grayscale(positives_folder)

        # # Plot the average pixel distributions of both classes on the same plot
        # plt.plot(average_distribution_negatives, label='Negatives')
        # plt.plot(average_distribution_positives, label='Positives')
        # plt.xlabel('Pixel Intensity')
        # plt.ylabel('Frequency')
        # plt.title('Avg. Pixel Distribution - Simplified Dataset')
        # plt.legend()
        # plt.savefig('data/SFHDataset/analysis/frequency_analysis_simplified.pdf', format='pdf')
        # plt.close()

    elif selected_domain == 'rgb':
        # Provide the paths to the folders containing videos for each class
        negatives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/0'
        positives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/1'

        # Compute the average pixel distribution for each class
        average_distribution_r_negatives, average_distribution_g_negatives, average_distribution_b_negatives = compute_average_distribution_rgb(negatives_folder)
        average_distribution_r_positives, average_distribution_g_positives, average_distribution_b_positives = compute_average_distribution_rgb(positives_folder)
        # Create subplots for R, G, and B channels
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Plot the average pixel distributions for each class in RGB domain
        axs[0].plot(average_distribution_r_negatives, color='red', label='Negatives')
        axs[0].plot(average_distribution_r_positives, color='blue', label='Positives')
        axs[0].set_xlabel('Pixel Intensity')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Average Pixel Distribution (Red Channel) - Original')
        axs[0].legend()

        axs[1].plot(average_distribution_g_negatives, color='green', label='Negatives')
        axs[1].plot(average_distribution_g_positives, color='purple', label='Positives')
        axs[1].set_xlabel('Pixel Intensity')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Average Pixel Distribution (Green Channel) - Original')
        axs[1].legend()

        axs[2].plot(average_distribution_b_negatives, color='blue', label='Negatives')
        axs[2].plot(average_distribution_b_positives, color='lightblue', label='Positives')
        axs[2].set_xlabel('Pixel Intensity')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Average Pixel Distribution (Blue Channel) - Original')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig('data/SFHDataset/analysis/frequency_analysis_rgb_new_dataset.pdf', format='pdf')
        plt.close()

        # Provide the paths to the folders containing videos for each class
        # negatives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/0'
        # positives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/1'

        # # Compute the average pixel distribution for each class
        # average_distribution_r_negatives, average_distribution_g_negatives, average_distribution_b_negatives = compute_average_distribution_rgb(negatives_folder)
        # average_distribution_r_positives, average_distribution_g_positives, average_distribution_b_positives = compute_average_distribution_rgb(positives_folder)
        # # Create subplots for R, G, and B channels
        # fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # # Plot the average pixel distributions for each class in RGB domain
        # axs[0].plot(average_distribution_r_negatives, color='red', label='Negatives')
        # axs[0].plot(average_distribution_r_positives, color='blue', label='Positives')
        # axs[0].set_xlabel('Pixel Intensity')
        # axs[0].set_ylabel('Frequency')
        # axs[0].set_title('Average Pixel Distribution (Red Channel) - Simplified')
        # axs[0].legend()

        # axs[1].plot(average_distribution_g_negatives, color='green', label='Negatives')
        # axs[1].plot(average_distribution_g_positives, color='purple', label='Positives')
        # axs[1].set_xlabel('Pixel Intensity')
        # axs[1].set_ylabel('Frequency')
        # axs[1].set_title('Average Pixel Distribution (Green Channel) - Simplified')
        # axs[1].legend()

        # axs[2].plot(average_distribution_b_negatives, color='blue', label='Negatives')
        # axs[2].plot(average_distribution_b_positives, color='lightblue', label='Positives')
        # axs[2].set_xlabel('Pixel Intensity')
        # axs[2].set_ylabel('Frequency')
        # axs[2].set_title('Average Pixel Distribution (Blue Channel) - Simplified')
        # axs[2].legend()

        # plt.tight_layout()
        # plt.savefig('data/SFHDataset/analysis/frequency_analysis_rgb_simplified.pdf', format='pdf')
        # plt.close()
