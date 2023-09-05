import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw left hand connections
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw right hand connections

    if results.multi_hand_landmarks:  # if there is at least one hand
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
                # mp_drawing.DrawingSpec(color=(121, 22,  76), thickness=2, circle_radius=4),
                # mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )


# results.multi_hand_landmarks[0].landmark
def extract_keypoints(results):
    h = (
        np.array(
            [[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]
        ).flatten()
        if results.multi_hand_landmarks
        else np.zeros(21 * 3)
    )  # se non trovi la mano, mettici tutti 0, 3 coordinate per 21 joints
    return h  # np.concatenate([pose, face, lh, rh]) questo potrà essere utile per gestire più mani


def main():
    # https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()


if __name__ == "__main__":
    main()
