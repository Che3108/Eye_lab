#!/usr/bin/python3
import cv2
from nn_predict import paint_box

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = paint_box(img[32:416+32,112:416+112,:])
    cv2.imshow("camera", img)
    
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
cap.release()
cv2.destroyAllWindows()