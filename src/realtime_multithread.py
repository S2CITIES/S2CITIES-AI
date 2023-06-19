import cv2
import mediapipe as mp
import numpy as np
import time
from featureextractor import FeatureExtractor
import joblib
from multiprocessing import Process, Queue
import threading

import multiprocessing



# Load the random forest model
#model = joblib.load('random_forest_model.joblib')

# Set up Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up the feature extractor
#feature_extractor = FeatureExtractor()

# Set up the video capture
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 12)
print(cap.get(cv2.CAP_PROP_FPS))

# Set up the hyperparameters
interval = 1  # in seconds
window = 2.5  # in seconds

# Set up the time variables
start_time = time.time()
last_second = -1


frame_rate = 12
#prev = 0



# limitazioni: sto considerando una sola mano e che sia continua

def thread_extract_keypoints():

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
            break

def thread_predict():
    # Load the random forest model
    #model = joblib.load('random_forest_model.joblib')
    frames_to_next_prediction = interval * frame_rate

    while True:
        # Sleep for 1 second
        time.sleep(1)

        print('PORCO DIO')
        timeseries_full = len(timeseries) == frame_rate * window

        if timeseries_full and frames_to_next_prediction == 0: # the timeseries is full
            # TODO
            # fai estrazione e prediction qui
            # Extract features from the timeseries
            #features = feature_extractor.extract_features(timeseries_2_5s)

            # Predict the output using the random forest model
            #output = model.predict(features.reshape(1, -1))[0]

            # Print the output to the console
            #print(output)
            print('prediction')

            # dopo aver estratto, resetta frames_to_next_prediction
            frames_to_next_prediction = interval * frame_rate
        elif timeseries_full and frames_to_next_prediction > 0:
            frames_to_next_prediction -= 1

if __name__ == '__main__':

    from multiprocessing import shared_memory
    timeseries= []
    A = shared_memory.ShareableList(["timeseries"])

    extract_process = Process(name='extract', target=thread_extract_keypoints)
    predict_process = Process(name='predict', target=thread_predict)

    # execute both processes at the same time
    # without waiting that each of their While loops ends
    extract_process.start()
    predict_process.start()

    # you must always close and unlink shared_memory
    A.shm.close()
    A.shm.unlink()

    # Release the video capture
    cap.release()
