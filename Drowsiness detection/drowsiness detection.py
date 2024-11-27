import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize Pygame mixer for sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade classifiers for face and eyes detection
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Labels for eyes status
lbl = ['Close', 'Open']

# Load the pre-trained CNN model
model = load_model('models/cnncat2.h5', compile=False)

path = os.getcwd()

# Initialize the webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes in the frame
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a black rectangle for score display area
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    rpred_class = [99]  # Initialize predictions
    lpred_class = [99]

    # Check if right eye is detected
    if len(right_eye) > 0:
        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)

            # Predict eye status (open or closed)
            rpred = model.predict(r_eye)
            rpred_class = np.argmax(rpred, axis=1)
            print(f"Right Eye Prediction: {rpred_class}")  # Debug print
            break  # Only need the first right eye detected

    # Check if left eye is detected
    if len(left_eye) > 0:
        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)

            # Predict eye status (open or closed)
            lpred = model.predict(l_eye)
            lpred_class = np.argmax(lpred, axis=1)
            print(f"Left Eye Prediction: {lpred_class}")  # Debug print
            break  # Only need the first left eye detected

    # Check both eyes' statuses and update score
    if rpred_class[0] == 0 and lpred_class[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure score doesn't go below 0
    score = max(0, score)
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score exceeds a threshold (indicating sleepiness)
    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        if not mixer.get_busy():  # Check if sound is not playing
            sound.play()
        
        # Flash red rectangle to indicate the alarm is triggered
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
