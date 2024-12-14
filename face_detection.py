import cv2


face_classifier = cv2.CascadeClassifier('/home/vasiliki/Desktop/test/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read() #read first frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any faces detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

   
    cv2.imshow('faces', frame)

    if cv2.waitKey(1) ==13: 
        break

cap.release()
cv2.destroyAllWindows()


