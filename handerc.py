import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

tipIds = [4, 8, 12, 16, 20]  # Landmark indices for the tips of fingers

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Finger counting
            fingers = []
            # Thumb
            if handLms.landmark[tipIds[0]].y < handLms.landmark[tipIds[0] - 1].y:
                fingers.append(1)
            else:
                fingers.append(0)
            # Other fingers
            for id in range(1, 5):
                if handLms.landmark[tipIds[id]].y < handLms.landmark[tipIds[id] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = sum(fingers)
            cv2.putText(img, f'Fingers: {total_fingers}', (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
