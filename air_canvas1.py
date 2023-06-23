import cv2
import mediapipe as mp
import numpy as np
import math

canvas = None

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

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_index, y_index = int(index_finger_landmark.x * image.shape[1]), int(index_finger_landmark.y * image.shape[0])
            z_index = int(index_finger_landmark.z*1000)

            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            x_thumb, y_thumb = int(thumb_landmark.x * image.shape[1]), int(thumb_landmark.y * image.shape[0])
            z_thumb = int(thumb_landmark.z*1000)

            distance = dist(x_index, y_index, z_index, x_thumb, y_thumb, z_thumb)

            if prev_x is not None and prev_y is not None:
                cv2.line(image, (prev_x, prev_y), (x_index, y_index), (0, 0, 0), thickness=2)

                if distance>40:
                    lines.append([(prev_x, prev_y), (x_index, y_index)])
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), (0, 0, 0), thickness=2)

            prev_x, prev_y = x_index, y_index
            

    if canvas is None:
        canvas = np.ones_like(image) * 255

    for line in lines:
        cv2.line(image, line[0], line[1], (0, 0, 0), thickness=2)

    cv2.imshow("Hand-Tracking", image)
    cv2.imshow("Air-Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = None
        prev_x, prev_y = None, None

cap.release()
cv2.destroyAllWindows()
