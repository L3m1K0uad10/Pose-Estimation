import time

import cv2
import mediapipe as mp 



mpDraw = mp.solutions.drawing_utils # creating object of drawing
mpPose = mp.solutions.pose # creating object of pose
pose = mpPose.Pose() # creating object of pose

cap = cv2.VideoCapture("Pose-Estimation/videos/6.mp4")
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting BGR to RGB
    results = pose.process(imgRGB) # processing the image
    #print(results.pose_landmarks) # printing the landmarks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # drawing the landmarks
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm) 
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)