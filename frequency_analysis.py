import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_video(video_path):
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

def compute_average_distribution(video_folder):
    videos = os.listdir(video_folder)
    total_videos = len(videos)
    average_distribution = np.zeros((256,), dtype=float)

    for video in videos:
        video_path = os.path.join(video_folder, video)
        pixel_distribution = analyze_video(video_path)
        average_distribution = np.add(average_distribution, pixel_distribution)

    average_distribution /= total_videos

    return average_distribution

if __name__ == '__main__':
    # Provide the paths to the folders containing videos for each class
    negatives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_ratio1_224x224/0'
    positives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_ratio1_224x224/1'

    # Compute the average pixel distribution for each class
    average_distribution_negatives = compute_average_distribution(negatives_folder)
    average_distribution_positives = compute_average_distribution(positives_folder)

    # Plot the average pixel distributions of both classes on the same plot
    plt.plot(average_distribution_negatives, label='Negatives')
    plt.plot(average_distribution_positives, label='Positives')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Avg. Pixel Distribution - Original Dataset')
    plt.legend()
    plt.savefig('data/SFHDataset/analysis/frequency_analysis_original.pdf', format='pdf')
    plt.close()

    negatives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_simplified_ratio1_224x224/0'
    positives_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_simplified_ratio1_224x224/1'

    # Compute the average pixel distribution for each class
    average_distribution_negatives = compute_average_distribution(negatives_folder)
    average_distribution_positives = compute_average_distribution(positives_folder)

    # Plot the average pixel distributions of both classes on the same plot
    plt.plot(average_distribution_negatives, label='Negatives')
    plt.plot(average_distribution_positives, label='Positives')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Avg. Pixel Distribution - Simplified Dataset')
    plt.legend()
    plt.savefig('data/SFHDataset/analysis/frequency_analysis_simplified.pdf', format='pdf')
    plt.close()