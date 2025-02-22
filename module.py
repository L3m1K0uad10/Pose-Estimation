import time

import cv2
import mediapipe as mp 



class PoseDetector():
    def __init__(self, mode = False, up_body = False, smooth = True, model_complexity = 1, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.up_body = up_body
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils 
        self.mpPose = mp.solutions.pose 
        self.pose = self.mpPose.Pose(
            self.mode, 
            self.up_body, 
            self.model_complexity,
            self.smooth, 
            self.detection_confidence, 
            self.tracking_confidence
        )

    def find_pose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.pose.process(imgRGB) 

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, 
                    self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
    
        return img
    
    def find_position(self, img, draw = True):
        lm_list = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm) 
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        
        return lm_list
    
    def draw_landmark(self, img, lm_list, lm_id):
        cv2.circle(img, (lm_list[lm_id][1], lm_list[lm_id][2]), 15, (255, 0, 0), cv2.FILLED)


def main():
    cap = cv2.VideoCapture("Pose-Estimation/videos/3.mp4")
    pTime = 0

    detector = PoseDetector()

    while True:
        success, img = cap.read() 

        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw = False)
        detector.draw_landmark(img, lm_list, 14)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(1)



if __name__ == "__main__":
    main()