import streamlit as st
import cv2
import mediapipe as mp
import keyboard as kb
import numpy as np
import math
from PIL import ImageColor

st.set_page_config('Air-Canvas')
st.title('Air-Canvas')

run = st.checkbox('Run')
FRAME = st.image([])
CANVAS = st.image([])

tools = ['Freehand', 'Line', 'Circle', 'Rectangle']

freehand = []
line = []
circle = []
rectangle = []
active = []
index = [0]

# if 'key' not in st.session_state:
#     st.session_state.key = None
# else:
#     active = st.session_state.key

if 'backup' not in st.session_state:
    st.session_state.backup = []
else:
    freehand = st.session_state.backup[0]
    line = st.session_state.backup[1]
    circle = st.session_state.backup[2]
    rectangle = st.session_state.backup[3]
    active = st.session_state.backup[4]
    index = st.session_state.backup[5]

def dist(x1, y1, x2, y2):
    d = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    return int(d)

def clear_all():
    freehand.clear()
    line.clear()
    circle.clear()
    rectangle.clear()

def Undo():
   l = len(st.session_state.backup[0])
    if active[-1] == 0 and len(index) > 1:
        for i in range(l-index[-2]):
            st.session_state.backup[0].pop(index[-2]) 
            
    elif active[-1] == 1 and st.session_state.backup[1] != []:
        st.session_state.backup[1].pop()
    elif active[-1] == 2 and st.session_state.backup[2] != []:
        st.session_state.backup[2].pop()
    elif active[-1] == 3 and st.session_state.backup[3] != []:
        st.session_state.backup[3].pop()
    
    if st.session_state.backup[4] != []:
        st.session_state.backup[4].pop()
    if st.session_state.backup[5] != [0]:
        st.session_state.backup[5].pop()


# Sidebar contents
with st.sidebar:
    tool = st.selectbox(label='DRAWING TOOL', options=tools, index=0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  )
    brush_color = ImageColor.getrgb(st.color_picker(label='BRUSH COLOR'))
    thickness = st.slider(label='BRUSH THICKNESS', min_value=1, max_value=10, value=3)
    canvas_color = ImageColor.getrgb(st.color_picker(label='CANVAS COLOR', value='#FFFFFF'))
    clear = st.button('CLEAR', on_click=clear_all)
    # col1, col2 = st.columns(2)
    # with col1:
    undo = st.button('UNDO', on_click=Undo)
    # with col2:
    #     redo = st.button('REDO', on_click=Redo)


canvas = None

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

prev_x, prev_y = None, None

cap = cv2.VideoCapture(0)
while run:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    draw = kb.is_pressed('shift')

    results = hands.process(image)
    i = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_index, y_index = int(index_finger_landmark.x * image.shape[1]), int(index_finger_landmark.y * image.shape[0])
            z_index = int(index_finger_landmark.z*1000)

            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            x_thumb, y_thumb = int(thumb_landmark.x * image.shape[1]), int(thumb_landmark.y * image.shape[0])
            z_thumb = int(thumb_landmark.z*1000)

            if prev_x is not None and prev_y is not None:
                cv2.circle(image, (x_index, y_index), 2*thickness, brush_color, -1, lineType=4)
                if tool == tools[0]: 
                    if draw:
                        freehand.append(((prev_x, prev_y), (x_index, y_index), brush_color, 2*thickness, 4))
                        i = len(freehand) - 1
                    else:
                        if i > index[-1]:
                            index.append(i)
                            active.append(0

                elif tool == tools[1]:
                    cv2.circle(image, (x_thumb, y_thumb), 2*thickness, brush_color, -1, lineType=4)
                    cv2.line(image, (x_index, y_index), (x_thumb, y_thumb), brush_color, 2*thickness, lineType=4)
                    if draw:
                        line.append(((x_index, y_index), (x_thumb, y_thumb), brush_color, 2*thickness, 4))
                        active.append(1)
                
                elif tool == tools[2]:
                    cv2.circle(image, (x_thumb, y_thumb), 2*thickness, brush_color, -1, lineType=4)
                    cv2.circle(image, (int((x_index + x_thumb)/2), int((y_index + y_thumb)/2)), int(dist(x_index, y_index, x_thumb, y_thumb)/2), brush_color, 2*thickness, lineType=4)
                    if draw:
                        circle.append(((x_index, y_index), (x_thumb, y_thumb), brush_color, 2*thickness, 4))
                        active.append(2)

                elif tool == tools[3]:
                    cv2.circle(image, (x_thumb, y_thumb), 2*thickness, brush_color, -1, lineType=4)
                    cv2.rectangle(image, (x_index, y_index), (x_thumb, y_thumb), brush_color, 2*thickness, lineType=4)
                    if draw:
                        rectangle.append(((x_index, y_index), (x_thumb, y_thumb), brush_color, 2*thickness, 4))
                        active.append(3)

            prev_x, prev_y = x_index, y_index

    if canvas is None:
        canvas = np.ones_like(image) * 255  
        canvas[:, :, 0] = canvas_color[0]  
        canvas[:, :, 1] = canvas_color[1]
        canvas[:, :, 2] = canvas_color[2]

    
    for point in freehand:
        cv2.line(image, point[0], point[1], point[2], point[3], lineType=point[4])
        cv2.line(canvas, point[0], point[1], point[2], point[3], lineType=point[4])

   
    for point in line:
        cv2.line(image, point[0], point[1], point[2], point[3], lineType=point[4])
        cv2.line(canvas, point[0], point[1], point[2], point[3], lineType=point[4])

   
    for point in circle:
        cv2.circle(image, (int((point[0][0] + point[1][0])/2), int((point[0][1] + point[1][1])/2)), int((dist(point[0][0], point[0][1], point[1][0], point[1][1]))/2), point[2], point[3], lineType=point[4])
        cv2.circle(canvas, (int((point[0][0] + point[1][0])/2), int((point[0][1] + point[1][1])/2)), int((dist(point[0][0], point[0][1], point[1][0], point[1][1]))/2), point[2], point[3], lineType=point[4])

   
    for point in rectangle:
        cv2.rectangle(image, point[0], point[1], point[2], point[3], lineType=point[4])
        cv2.rectangle(canvas, point[0], point[1], point[2], point[3], lineType=point[4])


    FRAME.image(image)
    CANVAS.image(canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') or clear:
        canvas = None
        lines = []
        prev_x, prev_y = None, None

st.session_state.backup = [freehand , line, circle, rectangle, active, index]

cap.release()



