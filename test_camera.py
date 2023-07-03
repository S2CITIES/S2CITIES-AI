import cv2
import time
cap = cv2.VideoCapture(0)

while True:
    time.sleep(0.1)
    ret, frame = cap.read()
    print(ret)
    cv2.imshow('Camera', frame)