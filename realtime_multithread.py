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

import argparse
import threading
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from models.model_mpkpts import Model
from src.keypointsextractor import KeypointsExtractor

# Set up the feature extractor
#feature_extractor = FeatureExtractor()

# TODO
# limitazioni:
# - sto considerando una sola mano
# - e che sia continua (ma ok)
def thread_extract_keypoints():

    # Set up the global variables
    global timeseries

    # Set up the video capture from the webcam
    cap = cv2.VideoCapture(0)

    # Set up Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Set up the hyperparameters
    interval = 1                   # in seconds
    window = 2.5                   # in seconds
    prev = 0                       # in seconds
    frames_to_next_prediction = 0  # in frames
    frame_rate = 12                # in frames per second

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

        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the hand landmarks using Mediapipe
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            results = hands.process(frame_rgb)

            # If there is exactly one hand
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:

                # Extract the keypoints from the hand landmarks
                # TODO: implement the custom extractor
                keypoints = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
                # If the timeseries is not full, add the keypoints to it
                if len(timeseries) < frame_rate * window:
                    timeseries.append(keypoints)
                else:
                    # If the timeseries is full, remove the first element
                    # and add the new keypoints to the end (i.e shift the timeseries)
                    timeseries[:-1] = timeseries[1:]
                    timeseries[-1] = keypoints
            elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                print("Multiple hands detected. Ignoring.")
            elif not results.multi_hand_landmarks:
                # if there are not regognized keypoints, reset the timeseries
                timeseries = []

                # Set the frames_to_next_prediction to zero so that when
                # the timeseries gets full again, it can immediately perform prediction
                frames_to_next_prediction = 0

        # Flag if the timeseries is full
        timeseries_full = len(timeseries) == frame_rate * window

        # Check whether to trigger prediction
        if timeseries_full and frames_to_next_prediction == 0:
            predict_event.set()
            frames_to_next_prediction = interval * frame_rate
        elif timeseries_full and frames_to_next_prediction > 0:
            frames_to_next_prediction -= 1

        # TODO
        # si può implementare un sistema per non dover aspettare
        # per forza che la timeseries sia full, ma consentendo anche
        # lunghezze minori da riempire con zeri o NAN

        # Print the length of the timeseries to the console for debugging
        #print("Timeseries length:", len(timeseries))

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow('Hand Keypoints', frame)

        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Perform the last prediction so that the thread can stop
            predict_event.set()

            # Trigger the stop event that will stop the predict thread
            stop_event.set()

            # Release the video capture
            cap.release()
            break


def thread_predict(
        stop_event,
        predict_event,
        training_results,
        model_choice,
        threshold,
        tsfresh_parameters,
        scaler,
        final_features,
        ):

    # Load the random forest model
    model = Model(
        training_results=training_results,
        model_choice=model_choice,
        tsfresh_parameters=tsfresh_parameters,
        scaler=scaler,
        final_features=final_features,
        )

    while not stop_event.is_set():

        predict_event.wait()

        # Predict the output using the model
        proba = model.predict(timeseries)

        output = (proba[:,1] >= threshold).astype(bool)

        print(f"Predicted output: {output} [{proba[:,1]}]")

        # Reset the predict event
        predict_event.clear()

if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_results',
                        type=str,
                        required=True,
                        help='Path to the training results file')
    parser.add_argument('--tsfresh_parameters',
                        type=str,
                        required=True,
                        help='Path to the tsfresh parameters file')
    parser.add_argument('--scaler',
                        type=str,
                        required=True,
                        help='Path to the scaler file')
    parser.add_argument('--model_choice',
                        type=str,
                        choices=['RF', 'SVM', 'LR'],
                        required=True,
                        help='Name of the model to load')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Threshold for the classification')
    parser.add_argument('--final_features',
                        type=str,
                        required=True,
                        help='Path to the final features file')
    args = parser.parse_args()

    # Set up the timeseries
    timeseries = []

    # Create the stop event
    stop_event = threading.Event()

    # Create the predict event
    predict_event = threading.Event()

    # Start the predict thread
    predict_process = threading.Thread(
        name='predict',
        target=thread_predict,
        args=(
            stop_event,
            predict_event,
            args.training_results,
            args.model_choice,
            args.threshold,
            args.tsfresh_parameters,
            args.scaler,
            args.final_features,
            )
        )
    predict_process.start()

    # Start the extract keypoints thread which is the main thread
    thread_extract_keypoints()
