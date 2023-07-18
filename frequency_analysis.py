import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Provide the path to your video file
video_path = 'ffmpeg/0/vid_00002_00003.MOV'

# Analyze the video and get the pixel frequency distribution
pixel_distribution = analyze_video(video_path)

# Plot the pixel frequency distribution
plt.plot(pixel_distribution)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Frequency Distribution')
plt.show()
