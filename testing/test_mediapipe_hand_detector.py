import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# video_path = "path_to_video_file.mp4"
cap = cv2.VideoCapture("dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES/1/vid_00001_00001.MOV")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = y_min = float("inf")
            x_max = y_max = float("-inf")
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Define the desired aspect ratio
            aspect_ratio = 1  # Example: 16:9 aspect ratio

            # Calculate the center coordinates of the hand landmarks
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            # Calculate the width and height of the bounding box
            width = max(x_max - x_min, (y_max - y_min) * aspect_ratio)
            height = max(y_max - y_min, (x_max - x_min) / aspect_ratio)

            # Calculate the new bounding box coordinates based on the center, width, and height
            x_min = int(x_center - width / 2)
            x_max = int(x_center + width / 2)
            y_min = int(y_center - height / 2)
            y_max = int(y_center + height / 2)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    

