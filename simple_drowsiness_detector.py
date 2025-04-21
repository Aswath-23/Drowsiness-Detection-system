import cv2
import time
import winsound
import numpy as np
import threading

def play_alarm():
    while alarm_playing:
        # Play system alarm
        winsound.PlaySound("SystemExclamation", winsound.SND_ASYNC)
        time.sleep(0.5)  # Wait before playing again
        # Play a different alarm sound
        winsound.Beep(1000, 500)  # High frequency beep
        time.sleep(0.5)

def main():
    global alarm_playing
    alarm_playing = False
    alarm_thread = None
    
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Load the eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Variables for drowsiness detection
    eyes_closed_time = 0
    last_eye_detection_time = time.time()
    alert_duration = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        
        if len(eyes) >= 2:  # If at least 2 eyes are detected
            eyes_closed_time = 0
            last_eye_detection_time = current_time
            if alarm_playing:
                alarm_playing = False
                if alarm_thread:
                    alarm_thread.join()  # Wait for alarm thread to finish
                winsound.PlaySound(None, winsound.SND_ASYNC)  # Stop alarm
                alert_duration = 0
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            eyes_closed_time = current_time - last_eye_detection_time
            
            # If eyes are closed for more than 2 seconds
            if eyes_closed_time > 2:
                if not alarm_playing:
                    alarm_playing = True
                    alarm_thread = threading.Thread(target=play_alarm)
                    alarm_thread.start()
                
                # Create a red overlay for alert
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Display warning with blinking effect
                alert_duration += 1
                if alert_duration % 10 < 5:  # Blink every 10 frames
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.putText(frame, "WAKE UP!", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    alarm_playing = False
    if alarm_thread:
        alarm_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 