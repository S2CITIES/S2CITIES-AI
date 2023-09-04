import json
import os
import random
import threading
import time

import cv2
#import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
#import torchvision.transforms as transforms
from PIL import Image

import transforms.spatial_transforms as SPtransforms
import transforms.temporal_transforms as TPtransforms
from build_models import build_model
from s2citiesAppSdk import import_zone_config, send_signal_for_help


def thread_collect_frames(hand_detector=True):
    while True:
        time.sleep(2.5)
        predict_event.set()
        
        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Perform the last prediction so that the thread can stop
            predict_event.set()
            # Trigger the stop event that will stop the predict thread
            stop_event.set()
            break


def thread_predict(stop_event, predict_event, alert_event, model, spatial_transform, threshold):

    global inference_steps
    results_queue =[0]*3
    ignore_predictions = 0

    while not stop_event.is_set():

        predict_event.wait()
        start_time = time.time()

        result = random.randint(0,1)
        inference_steps += 1

        end_time = time.time()

        print(f"--- Step {inference_steps} ---")
        print(f"Predicted output: {result}")
        print(f"Total inference time: {end_time-start_time}")
        
        # Alert if 2 out of 3 results are positive, then ignore the next 5 predictions
        results_queue.pop(0)
        results_queue.append(result)
        if len(results_queue) == 3 and results_queue.count(1) >= 2 and ignore_predictions == 0:
            alert_event.set()
            ignore_predictions = 20
        if ignore_predictions > 0:
            ignore_predictions -= 1

        # Reset the predict event
        predict_event.clear()


def thread_alert(stop_event, alert_event):

    global zone_config

    while True:
        alert_event.wait()
        start_time = time.time()

        if stop_event.is_set():
            break

        print("Sending alert..............................................................")

        # 2. Create a new Signal For Help Alert
        video_path = retrieve_video_path()

        new_created_alert = send_signal_for_help(zone_config, video_path)
        #new_created_alert = True

        if new_created_alert != None:
            print("Alert sent successfully!...................................................")

        end_time = time.time()
        print(f"Total alerting time: {end_time-start_time}")

        # Reset the predict event
        alert_event.clear()


def retrieve_video_path():
    # * a new alert has to be sent *
    
    # (properly retrieve the video filepath)
    # TODO: ADD LOGIC TO SAVE THE 10s TO SEND AND RETRIEVE FILEPATH
    video_path = "C:\\Users\\Dario\\Downloads\\VID_000000.mp4"

    return video_path


if __name__ == '__main__':
    # 1. import zone configuration data
    zone_config = import_zone_config("s2citiesAppSdk/s2cities-testzone-config.json")

    # Set up the frame queue (to push and pop frames)
    frame_queue = []
    inference_steps = 0
    print("Building gesture recognition model...")

    # PUT HERE STUFF TO CREATE REALTIME MODEL
    model = None
    spatial_transform = None
    threshold = 0.5

    # Create the stop event
    stop_event = threading.Event()

    # Create the predict event
    predict_event = threading.Event()

    # Create the alert event
    alert_event = threading.Event()

    # Start the predict thread
    predict_process = threading.Thread(name='predict', target=thread_predict, args=(stop_event, predict_event, alert_event, model, spatial_transform, threshold))
    predict_process.start()

    # Start the alert thread
    alert_process = threading.Thread(name='alert', target=thread_alert, args=(stop_event, alert_event))
    alert_process.start()

    # Start the extract keypoints thread which is the main thread
    thread_collect_frames(hand_detector=False)

    predict_process.join()
    alert_process.join()
