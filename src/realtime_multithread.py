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



import cv2
import mediapipe as mp
import numpy as np
import time
from featureextractor import FeatureExtractor
import joblib
from multiprocessing import Process, Queue
import threading

import multiprocessing

from threading import Thread


# Load the random forest model
#model = joblib.load('random_forest_model.joblib')


# Set up the feature extractor
#feature_extractor = FeatureExtractor()



# limitazioni: sto considerando una sola mano e che sia continua

def thread_extract_keypoints():

    global timeseries

    

    # Set up the video capture from a video file
    #cap = cv2.VideoCapture('/Users/teo/Documents/GitHub/S2CITIES-AI/src/dataset_creation/4_videos_labeled/1/vid_00001_00001.MOV')
    cap = cv2.VideoCapture(0)
    frame_rate = 12

    # Set up Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands


    # Set up the hyperparameters
    interval = 1  # in seconds
    window = 2.5  # in seconds

    frames_to_next_prediction = 0

    prev = 0
    while True:

        # Read a frame from the video capture
        ret, frame = cap.read()


        # Implement logit to limit the frame rate
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
                keypoints = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() # se non trovi la mano, mettici tutti 0, 3 coordinate per 21 joints
                # If the timeseries is not full, add the keypoints to it
                if len(timeseries) < frame_rate * window:
                    timeseries.append(keypoints)
                else:
                    # If the timeseries is full, remove the first element and add the new keypoints to the end
                    timeseries[:-1] = timeseries[1:]
                    timeseries[-1] = keypoints
            elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                print("Multiple hands detected. Fuck you")
            elif not results.multi_hand_landmarks:
                #print("NO RESULT")
           #    # if there are not regognized keypoints, reset the timeseries
                timeseries = []

                # Set the frames_to_next_prediction to zero so that when
                # the timeseries gets full, it can immediately perform prediction
                frames_to_next_prediction = 0
            
        timeseries_full = len(timeseries) == frame_rate * window

        if timeseries_full and frames_to_next_prediction == 0:
            predict_event.set()
            frames_to_next_prediction = interval * frame_rate
        elif timeseries_full and frames_to_next_prediction > 0:
            frames_to_next_prediction -= 1


        # TODO si può implementare un sistema per non dover aspettare per forza che la timeseries sia full, ma
        # consentendo anche lunghezze minori da riempire con zeri o NAN

        # Print the length of the timeseries to the console
        print("Timeseries length:", len(timeseries))

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow('Hand Keypoints', frame)

        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            predict_event.set() # perform the last prediction so that the thread can stop
            stop_event.set()
            break



def thread_predict(stop_event, predict_event):

    frame_rate = 12

    # Set up the hyperparameters
    interval = 1  # in seconds
    window = 2.5  # in seconds

    # Load the random forest model
    #model = joblib.load('random_forest_model.joblib')
    frames_to_next_prediction = interval * frame_rate

    while not stop_event.is_set():
        
        predict_event.wait()
        print('predict event set')
        print('running prediction')
        
        # Reset the predict event
        predict_event.clear()

        #timeseries_full = len(timeseries) == frame_rate * window

        #if predict_event: # the timeseries is full
            # TODO
            # fai estrazione e prediction qui
            # Extract features from the timeseries
            #features = feature_extractor.extract_features(timeseries_2_5s)

            # Predict the output using the random forest model
            #output = model.predict(features.reshape(1, -1))[0]

            # Print the output to the console
            #print(output)
            #print('prediction')

            # dopo aver estratto, resetta frames_to_next_prediction
            #frames_to_next_prediction = interval * frame_rate

        # Sleep for 1 second
        #time.sleep(1)

if __name__ == '__main__':

    # Set up the timeseries
    timeseries = []

    # create the stop event
    stop_event = threading.Event()
    predict_event = threading.Event()

    predict_process = Thread(name='predict', target=thread_predict, args=(stop_event,predict_event))
    predict_process.start()

    thread_extract_keypoints()