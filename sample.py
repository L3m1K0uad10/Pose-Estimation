import time

import cv2
import mediapipe as mp 

import module as md

cap = cv2.VideoCapture("Pose-Estimation/videos/3.mp4")
pTime = 0

detector = md.PoseDetector()

while True:
    success, img = cap.read() 

    img = detector.find_pose(img)
    lm_list = detector.find_position(img, draw = False)
    if len(lm_list) != 0:
        detector.draw_landmark(img, lm_list, 14)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)