import cv2
import time
import numpy as np 


car_classifier = cv2.CascadeClassifier('/home/vasiliki/Desktop/haarcascade_car.xml')

cap = cv2.VideoCapture('/home/vasiliki/Desktop/27260-362770008_small.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


while cap.isOpened():
    
    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.startWindowThread()
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) ==13: 
        break
        
cap.release()
cv2.destroyAllWindows()