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
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize

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


def thread_extract_keypoints():

    # Set up the global variables
    global frame_queue

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
    frame_rate = 12                # in frames per second

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        print(f"RET: {ret}")

        # Implement logit to limit the frame rate to frame_rate
        time_elapsed = time.time() - prev
        if not time_elapsed > 1./frame_rate:
            continue
        prev = time.time()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        # Convert the frame to RGB 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the aspect ratio of the frame to a target ratio of 1 by cropping
        frame_rgb = crop_frame(frame_rgb, height=height, width=width, aspect_ratio=aspect_ratio, target_ratio=1)

        # Resize the frame
        frame_rgb = cv2.resize(frame_rgb, (224, 224))

        # Push the frame to the frame queue
        frame_queue.append(frame_rgb)

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

        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Perform the last prediction so that the thread can stop
            predict_event.set()

            # Trigger the stop event that will stop the predict thread
            stop_event.set()

            # Release the video capture
            cap.release()
            break


def thread_predict(stop_event, predict_event, model, transform):

    while not stop_event.is_set():

        predict_event.wait()
        # # start_time = time.time()
        # video = torch.stack([transforms.ToTensor()(frame) for frame in frame_queue])
        # video = transform(video.permute(1, 0, 2, 3))
        # video = torch.unsqueeze(video, dim=0) # Add the batch dimension 

        # with torch.no_grad():
        #     # Predict the output using the model
        #     output = model(video)
        print("CIAO")
        # # end_time = time.time()
        # print(output)
        # output = torch.argmax(torch.softmax(output, dim=1), dim=1) # Take the prediction with the highest softmax score
        # print(f"Predicted output: {output}")
        # # print(f"Total inference time: {end_time-start_time}")
        # Reset the predict event
        predict_event.clear()

if __name__ == '__main__':

    # Set up the frame queue (to push and pop frames)
    frame_queue = []
    # model_type = 'mobilenetv2' # Make the model type an argument through argparse
    # model_path = 'models/saves/best_sfh_mobilenetv2.h5'
    # Load 3DCNN model
    # model = build_model(model_path=model_path, 
    #                     type=model_type, 
    #                    gpus=[0],           # Inference on a single GPU
    #                    num_classes=2,
    #                    finetune=False,     # Load the entire model as it is, no fine-tuning
    #                    state_dict=True)    # Only the state_dict of the model was saved
    # model.eval()                            # Set model to eval model

    # transform = transforms.Compose([
    #     UniformTemporalSubsample(num_samples=16, temporal_dim=-3),
    #    Normalize(mean=[0.4666, 0.4328, 0.3962], std=[0.2529, 0.2532, 0.2479]) 
    # ])

    # Create the stop event
    stop_event = threading.Event()

    # Create the predict event
    predict_event = threading.Event()

    # Start the predict thread
    # predict_process = threading.Thread(name='predict', target=thread_predict, args=(stop_event,predict_event, model, transform))
    # predict_process.start()

    # Start the extract keypoints thread which is the main thread
    thread_extract_keypoints()

    # predict_process.join()
