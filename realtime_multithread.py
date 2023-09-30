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
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import realtime_params as params
from mobileSDK.src.api import import_zone_config, send_signal_for_help
from models.model_mpkpts import Model


def thread_extract_keypoints():
    # Set up the global variables
    global timeseries, video_for_alert, alerts_sent

    # Set up the video capture from the webcam
    cap = cv2.VideoCapture(0)

    # Set up Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Set up the hyperparameters
    interval = params.interval

    prev = 0  # in seconds
    frames_to_next_prediction = 0  # in frames

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        if not ret:
            # Trigger the stop event that will stop the predict thread
            stop_event.set()

            # Perform the last prediction so that the thread can stop
            predict_event.set()

            # Perform the last alert so that the thread can stop
            alert_event.set()

            # Release the video capture
            cap.release()
            break

        # Implement logit to limit the frame rate to params.frame_rate
        time_elapsed = time.time() - prev
        if not time_elapsed > 1.0 / params.frame_rate:
            continue
        prev = time.time()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add frame to the queue to send in case of alert
        if len(video_for_alert) > params.frame_rate * params.video_alert_duration:
            video_for_alert.pop(0)
        video_for_alert.append(frame_rgb)

        # Detect the hand landmarks using Mediapipe
        with mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as hands:
            results = hands.process(frame_rgb)

            # If there is exactly one hand
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                # Extract the keypoints from the hand landmarks
                keypoints = np.array(
                    [
                        [res.x, res.y, res.z]
                        for res in results.multi_hand_landmarks[0].landmark
                    ]
                ).flatten()
                # If the timeseries is not full, add the keypoints to it
                if len(timeseries) < params.frame_rate * params.window:
                    timeseries.append(keypoints)
                else:
                    # If the timeseries is full, remove the first element
                    # and add the new keypoints to the end (i.e shift the timeseries)
                    timeseries[:-1] = timeseries[1:]
                    timeseries[-1] = keypoints
            elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                print("Multiple hands detected. Ignoring.")
                timeseries = []
                frames_to_next_prediction = 0
            elif not results.multi_hand_landmarks:
                # if there are not regognized keypoints, reset the timeseries
                timeseries = []

                # Set the frames_to_next_prediction to zero so that when
                # the timeseries gets full again, it can immediately perform prediction
                frames_to_next_prediction = 0

        # Flag if the timeseries is full
        timeseries_full = len(timeseries) == params.frame_rate * params.window

        # Check whether to trigger prediction
        if timeseries_full and frames_to_next_prediction == 0:
            predict_event.set()
            frames_to_next_prediction = interval * params.frame_rate
        elif timeseries_full and frames_to_next_prediction > 0:
            frames_to_next_prediction -= 1

        # TODO
        # si può implementare un sistema per non dover aspettare
        # per forza che la timeseries sia full, ma consentendo anche
        # lunghezze minori da riempire con zeri o NAN

        # Print the length of the timeseries to the console for debugging
        # print("Timeseries length:", len(timeseries))

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Show the frame
        cv2.imshow("Hand Keypoints", frame)

        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # Trigger the stop event that will stop the predict thread
            stop_event.set()

            # Perform the last prediction so that the thread can stop
            predict_event.set()

            # Perform the last alert so that the thread can stop
            alert_event.set()

            # Release the video capture
            cap.release()
            break


def thread_predict(
    stop_event,
    predict_event,
    alert_event,
    training_results,
    model_choice,
    threshold,
    tsfresh_parameters,
    scaler,
    final_features,
):
    results_queue = [0] * params.results_queue_length
    ignore_counter = 0

    # Load the model
    model = Model(
        training_results=training_results,
        model_choice=model_choice,
        tsfresh_parameters=tsfresh_parameters,
        scaler=scaler,
        final_features=final_features,
    )

    while not stop_event.is_set():
        predict_event.wait()

        # If the timeseries is not valid (i.e. it's the trigger before shutting down)
        # then skip the prediction
        timeseries_full = len(timeseries) == params.frame_rate * params.window
        if not timeseries_full:
            continue

        # Predict the output using the model
        proba = model.predict(timeseries)

        output = (proba[:, 1] >= threshold).astype(bool)

        print(f"Predicted output: {output} [{proba[:,1]}]")

        # Alert if params.results_queue_tolerance out of
        # params.results_queue_length results are positive, then ignore
        # the next params.num_predictions_ignore predictions
        results_queue.pop(0)
        results_queue.append(output)
        if (
            len(results_queue) == params.results_queue_length
            and results_queue.count(True) >= params.results_queue_tolerance
            and ignore_counter == 0
        ):
            alert_event.set()
            ignore_counter = params.num_predictions_ignore
        if ignore_counter > 0:
            ignore_counter -= 1

        # Reset the predict event
        predict_event.clear()


def thread_alert(stop_event, alert_event):
    global zone_config, alerts_sent

    while True:
        alert_event.wait()

        if stop_event.is_set():
            break

        # sleep for params.video_alert_post_duration seconds to include
        # a few extra seconds after the signal in the video sent
        time.sleep(params.video_alert_post_duration)

        start_time = time.time()

        print("Sending alert...")

        # Create a new Signal For Help Alert
        video_path = save_video(video_for_alert, params.frame_rate, alerts_sent)

        if video_path is None:
            continue

        new_created_alert = send_signal_for_help(zone_config, video_path)

        end_time = time.time()

        if new_created_alert is not None:
            print(
                f"Alert sent successfully!\nTotal alerting time: {end_time-start_time}"
            )
            alerts_sent += 1

        # Reset the alert event
        alert_event.clear()


def save_video(video, fps, n_alert):
    print("Saving video...")
    input = np.stack(video)
    print(f"Shape {input.shape}")

    # input = np.transpose(input, (1,2,3,0)) # Permuting to Tx(HxWxC)

    # input = np.uint8(input)
    video_path = f"alert_videos/alert_{n_alert}.mp4"

    writer = cv2.VideoWriter(
        filename=video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(int(input.shape[2]), int(input.shape[1])),
        isColor=True,
    )

    if writer.isOpened:
        for frame in input:
            frame_colored = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_colored)
        writer.release()
    else:
        print("Error saving the video!")
        return None

    return video_path


if __name__ == "__main__":
    # Import zone configuration data
    zone_config = import_zone_config("zone-config.json")

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_results",
        type=str,
        required=True,
        help="Path to the training results file",
    )
    parser.add_argument(
        "--tsfresh_parameters",
        type=str,
        required=True,
        help="Path to the tsfresh parameters file",
    )
    parser.add_argument(
        "--scaler", type=str, required=True, help="Path to the scaler file"
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        choices=["RF", "SVM", "LR"],
        required=True,
        help="Name of the model to load",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for the classification"
    )
    parser.add_argument(
        "--final_features",
        type=str,
        required=True,
        help="Path to the final features file",
    )
    args = parser.parse_args()

    # Create the alert_videos folder if it doesn't exist using pathlib
    Path("alert_videos").mkdir(parents=True, exist_ok=True)

    # Set up the timeseries
    timeseries = []
    video_for_alert = []
    alerts_sent = 0

    # Create the stop event
    stop_event = threading.Event()

    # Create the predict event
    predict_event = threading.Event()

    # Create the alert event
    alert_event = threading.Event()

    # Start the predict thread
    predict_process = threading.Thread(
        name="predict",
        target=thread_predict,
        args=(
            stop_event,
            predict_event,
            alert_event,
            args.training_results,
            args.model_choice,
            args.threshold,
            args.tsfresh_parameters,
            args.scaler,
            args.final_features,
        ),
    )
    predict_process.start()

    # Start the alert thread
    alert_process = threading.Thread(
        name="alert", target=thread_alert, args=(stop_event, alert_event)
    )
    alert_process.start()

    # Start the extract keypoints thread which is the main thread
    thread_extract_keypoints()

    # End threads
    predict_process.join()
    alert_process.join()
