import cv2
import mediapipe as mp
import numpy as np


canvas = None

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

prev_x, prev_y = None, None

cap = cv2.VideoCapture(0)
while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_landmark.x * image.shape[1]), int(index_finger_landmark.y * image.shape[0])

            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), thickness=2)
            prev_x, prev_y = x, y

    if canvas is None:
        canvas = np.ones_like(image) * 255

    cv2.imshow("Air Canvas", canvas)
    cv2.imshow("Hand Tracking", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = None
        prev_x, prev_y = None, None

cap.release()
cv2.destroyAllWindows()
