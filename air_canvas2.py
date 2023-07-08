import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

paintWindow = None
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def dist(x1, y1, z1, x2, y2, z2):
    d = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2) + math.pow(z2-z1, 2))
    return int(d)

prev_x, prev_y = None, None

lines = []

cap = cv2.VideoCapture(0)
while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    image.flags.writeable = True

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    image = cv2.rectangle(image, (40,1), (140,65), (0,0,0), 2)
    image = cv2.rectangle(image, (160,1), (255,65), (255,0,0), 2)
    image = cv2.rectangle(image, (275,1), (370,65), (0,255,0), 2)
    image = cv2.rectangle(image, (390,1), (485,65), (0,0,255), 2)
    image = cv2.rectangle(image, (505,1), (600,65), (0,255,255), 2)
    cv2.putText(image, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_index, y_index = int(index_finger_landmark.x * image.shape[1]), int(index_finger_landmark.y * image.shape[0])
            z_index = int(index_finger_landmark.z*1000)
            index_finger_landmark = (int(index_finger_landmark.x), int(index_finger_landmark.y))

            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            x_thumb, y_thumb = int(thumb_landmark.x * image.shape[1]), int(thumb_landmark.y * image.shape[0])
            z_thumb = int(thumb_landmark.z*1000)
            thumb_landmark = (int(thumb_landmark.x), int(thumb_landmark.y))

            distance = dist(x_index, y_index, z_index, x_thumb, y_thumb, z_thumb)

            if distance < 40:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            else:
                if y_index <= 65:
                    if 40 <= x_index <= 140: # Clear Button
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0

                        paintWindow[67:,:,:] = 255

                    elif 160 <= x_index <= 255:
                        colorIndex = 1 # Blue
                    elif 275 <= x_index <= 370:
                        colorIndex = 2 # Green
                    elif 390 <= x_index <= 485:
                        colorIndex = 3 # Red
                    elif 505 <= x_index <= 600:
                        colorIndex = 4 # Yellow

                else :
                    if colorIndex == 1:
                        bpoints[blue_index].appendleft((x_index, y_index))
                    elif colorIndex == 2:
                        gpoints[green_index].appendleft((x_index, y_index))
                    elif colorIndex == 3:
                        rpoints[red_index].appendleft((x_index, y_index))
                    elif colorIndex == 4:
                        ypoints[yellow_index].appendleft((x_index, y_index))


            points = [bpoints, gpoints, rpoints, ypoints]

            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.circle(image, (x_index, y_index), 3, colors[i], -1)
                        cv2.line(image, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)


    if paintWindow is None:
        paintWindow = np.ones_like(image) * 255

    cv2.imshow("Hand-Tracking", image)
    cv2.imshow("Air-Canvas", paintWindow)

    key = cv2.waitKey(1) 
    if key == ord('q'):
       break
    elif key == ord('c'):
       paintWindow = None
       prev_x, prev_y = None, None 
   

cap.release()
cv2.destroyAllWindows()
