"""
This module contains the FeatureExtractor class, which is used to extract the keypoints from a video.
"""

import cv2
import mediapipe as mp
import numpy as np

class KeypointsExtractor:
    def __init__(self, video_path, show_image=True):
        self.video_path = video_path
        self.show_image = show_image
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.NUM_FEATURES_PER_FRAME = 63

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks: # if there is at least one hand
            for num, hand in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                          self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                          self.mp_drawing_styles.get_default_hand_connections_style()
                                          #mp_drawing.DrawingSpec(color=(121, 22,  76), thickness=2, circle_radius=4),
                                          #mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

    def extract_keypoints(self, results):
        h = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(21*3) # se non trovi la mano, mettici tutti 0, 3 coordinate per 21 joints
        return h #np.concatenate([pose, face, lh, rh]) questo potrà essere utile per gestire più mani

    def extract_keypoints_from_video(self):
        # Open video file
        cap = cv2.VideoCapture(self.video_path)

        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a numpy array to store the keypoints
        keypoints = np.zeros((frame_count, self.NUM_FEATURES_PER_FRAME))

        # Set mediapipe model
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            for frame_num in range(frame_count):

                # Read feed
                ret, frame = cap.read()

                # Break if problem reading frame
                if not ret:
                    print("Error reading frame {}".format(frame_num + 1))
                    return None

                # Make detections
                image, results = self.mediapipe_detection(frame, hands)

                # Draw landmarks
                self.draw_landmarks(image, results)

                # Show to screen
                if self.show_image:
                    cv2.imshow('OpenCV Feed', image)

                # Check if excatly one hand is detected and save keypoints
                if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 1:
                    keypoints[frame_num] = self.extract_keypoints(results)
                # Otherwise, if no hand is detected, return None
                else:
                    return None

                # Break gracefully
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        return keypoints
