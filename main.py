import cv2
import pygame
import numpy as np

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file
audio_file = "alert.wav"  # Replace with the path to your audio file
pygame.mixer.music.load(audio_file)

# Load YOLOv3 model and classes (you can skip this part if you don't need YOLO)
# ... (Same as previous code)

cam = cv2.VideoCapture(0)  # Use camera index 0

oddness_threshold = 5000  # Adjust this value to control sensitivity to changes

while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    odd_event = False
    for c in contours:
        if cv2.contourArea(c) < oddness_threshold:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        odd_event = True
    
    if odd_event:
        pygame.mixer.music.play()  # Play the audio when something odd is detected

    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('Granny Cam', frame1)

# Stop and quit pygame mixer
pygame.mixer.music.stop()
pygame.mixer.quit()

cam.release()
cv2.destroyAllWindows()
