import numpy as np
import cv2
  
cap = cv2.VideoCapture('./dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224/1/vid_00006_00006.mp4')
  
# initializing subtractor 
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = cv2.VideoWriter("./test_background_remotion.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                cap.get(cv2.CAP_PROP_FPS),
                                (width, height),
                                isColor=False)

while(1):
    ret, frame = cap.read()         
  
    # applying on each frame
    if ret:
        fgmask = fgbg.apply(frame)  
    
        cv2.imshow('Background Remotion - Test', fgmask)
        output_video.write(fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
cap.release()
cv2.destroyAllWindows()