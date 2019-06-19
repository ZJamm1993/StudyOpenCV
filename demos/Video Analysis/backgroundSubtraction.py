import numpy as np
from cv2 import cv2

cap = cv2.VideoCapture('resource/video_test_p22.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while(1):
    ret, frame = cap.read()
    if ret == False:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
    fgmask = cv2.bitwise_and(fgmask,frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()