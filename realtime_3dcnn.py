"""
This file runs the realtime classification of hand gestures using the trained model.

Note: if you're getting an error
cv2.imshow() throwing "Unknown C++ exception from OpenCV code" only when threaded
under the issue https://github.com/opencv/opencv/issues/22602
they say:
- Use UI interaction functions from the "main" thread only.
- This is limitation of the platform, not OpenCV.

See also:
https://stackoverflow.com/questions/19790570/using-a-global-variable-with-a-thread
"""

import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from build_models import build_model
import transforms.spatial_transforms as SPtransforms
import transforms.temporal_transforms as TPtransforms
from PIL import Image
import os

def crop_frame(frame, height, width, aspect_ratio, target_ratio):

    if aspect_ratio > target_ratio:
        # Original video is wider, crop the sides
        target_width = int(height * target_ratio)
        crop_left = (width - target_width) // 2
        crop_right = width - target_width - crop_left
        crop_top, crop_bottom = 0, 0
    else:
        # Original video is taller, crop the top and bottom
        target_height = int(width / target_ratio)
        crop_top = (height - target_height) // 2
        crop_bottom = height - target_height - crop_top
        crop_left, crop_right = 0, 0

    return frame[crop_top:height - crop_bottom, crop_left:width - crop_right]


def thread_collect_frames(hand_detector=True):

    # Set up the global variables
    global frame_queue

    if hand_detector:
        # Initialize Hand Detector Model
        print("Initializing hand detector model...")
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

    # Set up the video capture from the webcam
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    aspect_ratio =  width / height

    print(f"Input video resolution: {(width, height)}")

    # Set up the hyperparameters
    interval = 1                   # in seconds
    window = 2.5                   # in seconds
    prev = 0                       # in seconds
    frames_to_next_prediction = 0  # in frames
    frame_rate = 30                # in frames per second

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Implement logit to limit the frame rate to frame_rate
        time_elapsed = time.time() - prev
        if not time_elapsed > 1./frame_rate:
            continue
        prev = time.time()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        if not hand_detector:
            converted_frame = crop_frame(frame, height=height, width=width, aspect_ratio=aspect_ratio, target_ratio=1)
            converted_frame = Image.fromarray(converted_frame)
            frame_queue.append(converted_frame)
        else:
            with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                results = hands.process(frame)

                # One hand (and exactly ONE hand) is detected
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:

                    mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                    # Convert the aspect ratio of the frame to a target ratio of 1 by cropping
                    converted_frame = crop_frame(frame, height=height, width=width, aspect_ratio=aspect_ratio, target_ratio=1)

                    # Convert frame to PIL Image
                    converted_frame = Image.fromarray(converted_frame)

                    # Push the frame to the frame queue
                    frame_queue.append(converted_frame)
                else:
                    # No hand was detected, restarting the acquisition process
                    frame_queue = []
                    frames_to_next_prediction = 0

        # print("Appending to the queue...")
        # Flag if the timeseries is full
        frame_queue_full = len(frame_queue) == int(frame_rate * window)

        if frame_queue_full:
            # If the timeseries is full, after appending we remove the first frame of the queue
            frame_queue.pop(0)
            # Then we check if it's time to make a prediction
            if frames_to_next_prediction == 0:
                predict_event.set()
                frames_to_next_prediction = interval * frame_rate
            elif frames_to_next_prediction > 0:
                # If not, just decrease the number of frames to collect until the next
                frames_to_next_prediction -= 1

        # print("Showing the frame...")
        cv2.imshow('frame', frame)
        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Perform the last prediction so that the thread can stop
            predict_event.set()

            # Trigger the stop event that will stop the predict thread
            stop_event.set()

            # Release the video capture
            cap.release()
            break


def thread_predict(stop_event, predict_event, model, spatial_transform, temporal_transform, threshold):

    global inference_steps

    while not stop_event.is_set():

        predict_event.wait()
        start_time = time.time()

        # Apply Temporal Transform
        n_frames = len(frame_queue)
        frame_indices = list(range(1, n_frames+1))
        frame_indices = temporal_transform(frame_indices)
        clip = [frame_queue[i-1] for i in frame_indices]

        # Apply Spatial Transform
        spatial_transform.randomize_parameters()
        clip = [spatial_transform(frame) for frame in clip]

        clip_tensor = torch.stack(clip, dim=0) # Tensor with shape TCHW
        clip_tensor = clip_tensor.permute(1, 0, 2, 3) # Tensor with shape CTHW
        clip_tensor = torch.unsqueeze(clip_tensor, dim=0) # Add the batch dimension

        with torch.no_grad():
            # Predict the output using the model
            output = model(clip_tensor)

        end_time = time.time()
        output = torch.softmax(output, dim=1)
        print(output)

        if output[0][1].item() > threshold:
            result = 1
        else:
            result = 0

        print(f"--- Step {inference_steps} ---")
        print(f"Predicted output: {result}")
        print(f"Total inference time: {end_time-start_time}")

        codec = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec (e.g., "mp4v", "XVID")
        output_file = os.path.join('real_time_test', f'test_output_step{inference_steps}_result{result}.mp4')  # Output video file name
        inference_steps += 1
        frame_size = (112, 112)  # Frame size (width, height)
        fps = 16/2.5

        video_writer = cv2.VideoWriter(output_file, codec, fps, frame_size)

        for frame in clip:
            np_img = frame.numpy()
            np_img = np.transpose(np_img, (1, 2, 0))
            np_img = np_img * 1.0
            np_img = np_img.astype(np.uint8)
            video_writer.write(np_img)

        video_writer.release()

        # Reset the predict event
        predict_event.clear()

if __name__ == '__main__':

    # Set up the frame queue (to push and pop frames)
    frame_queue = []
    inference_steps = 0
    print("Builing gesture recognition model...")
    model_type = 'mobilenetv2' # Make the model type an argument through argparse
    model_path = 'checkpoints/best_model_b32_SGD_mobilenetv2_dwn5_es8_random.h5'
    threshold = 0.8
    # Load 3DCNN model
    model = build_model(model_path=model_path,
                        type=model_type,
                       gpus=[0],           # Inference on a single GPU
                       num_classes=2,
                       finetune=False,     # Load the entire model as it is, no fine-tuning
                       state_dict=True)    # Only the state_dict of the model was saved
    model.eval()                           # Set model to eval model

    # Initialize spatial and temporal transforms (test versions)
    spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(112),
        SPtransforms.CornerCrop(112, crop_position='c'), # Central Crop in Test
        SPtransforms.ToTensor(1.0),
        SPtransforms.Normalize(mean=[
            105.46717834472656,
            115.16809844970703,
            125.67674255371094
        ], std=[
            60.821353912353516,
            62.009315490722656,
            62.135868072509766
        ])
    ])

    temporal_transform = TPtransforms.TemporalRandomCrop(16, 5)

    # Create the stop event
    stop_event = threading.Event()

    # Create the predict event
    predict_event = threading.Event()

    # Start the predict thread
    predict_process = threading.Thread(name='predict', target=thread_predict, args=(stop_event,predict_event, model, spatial_transform, temporal_transform, threshold))
    predict_process.start()

    # Start the extract keypoints thread which is the main thread
    thread_collect_frames(hand_detector=False)

    predict_process.join()
