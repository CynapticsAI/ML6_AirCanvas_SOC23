import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import ImageColor

st.set_page_config('Air-Canvas')
st.title('Air-Canvas')

run = st.checkbox('Run')
FRAME = st.image([])
CANVAS = st.image([])

tools = ['Freehand', 'Line', 'Circle', 'Rectangle']

freehand, line, circle, rectangle, active, index = [], [], [], [], [], [0]

# Session state
if 'backup' not in st.session_state:
    st.session_state.backup = [[], [], [], [], [], [0]]
else:
    freehand , line, circle, rectangle, active, index = st.session_state.backup


def dist(x1, y1, x2, y2):
    d = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    return int(d)

def clear_all():
    freehand.clear()
    line.clear()
    circle.clear()
    rectangle.clear()

def Undo():
    if active != []:
        if active[-1] == 0 and len(index) > 1:
            if len(index) == 2:
                st.session_state.backup[0] = st.session_state.backup[0][:index[-2]]
            else:
                st.session_state.backup[0] = st.session_state.backup[0][:index[-2]+1]
            if st.session_state.backup[5] != [0]:
                st.session_state.backup[5].pop()
        elif active[-1] == 1 and st.session_state.backup[1] != []:
            st.session_state.backup[1].pop()
        elif active[-1] == 2 and st.session_state.backup[2] != []:
            st.session_state.backup[2].pop()
        elif active[-1] == 3 and st.session_state.backup[3] != []:
            st.session_state.backup[3].pop()
        
        if st.session_state.backup[4] != []:
            st.session_state.backup[4].pop()


# Sidebar contents
with st.sidebar:
    tool = st.selectbox(label='DRAWING TOOL', options=tools, index=0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  )
    brush_color = ImageColor.getrgb(st.color_picker(label='BRUSH COLOR'))
    thickness = st.slider(label='BRUSH THICKNESS', min_value=1, max_value=10, value=3)
    canvas_color = ImageColor.getrgb(st.color_picker(label='CANVAS COLOR', value='#FFFFFF'))
    clear = st.button('CLEAR', on_click=clear_all)
    undo = st.button('UNDO', on_click=Undo)
    capture_canvas = st.button('CAPTURE CANVAS')

canvas = None
captur_done = False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

prev_x, prev_y = None, None

i = 0

x_start, y_start, x_end, y_end = None, None, None, None

cap = cv2.VideoCapture(0)

while run:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    draw = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_index, y_index = int(index_finger_landmark.x * image.shape[1]), int(index_finger_landmark.y * image.shape[0])

            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            x_thumb, y_thumb = int(thumb_landmark.x * image.shape[1]), int(thumb_landmark.y * image.shape[0])
            
            if dist(x_index, y_index, x_thumb, y_thumb)>35:
                draw = True
                
            if prev_x is not None and prev_y is not None:
                cv2.circle(image, (x_index, y_index), 2*thickness, brush_color, -1, lineType=4)
                if tool == tools[0]: 
                    if draw:
                        freehand.append(((prev_x, prev_y), (x_index, y_index), brush_color, 2*thickness, 4))
                        i = len(freehand) - 1
                    else:
                        if i > index[-1]:
                            index.append(i)
                            active.append(0)

                elif tool == tools[1]:
                    cv2.circle(image, (x_thumb, y_thumb), 2*thickness, brush_color, -1, lineType=4)
                    if  draw:
                        if ((x_end != x_start) or (y_end != y_start)):
                            line.append(((x_start, y_start), (x_end, y_end), brush_color, 2*thickness, 4))
                            active.append(1)
                        x_start, y_start = x_index, y_index
                        x_end, y_end = x_index, y_index
                    if not draw:
                        x_end, y_end= x_index, y_index
                        if x_start is not None:
                            cv2.line(image, (x_start, y_start), (x_end, y_end), brush_color, 2*thickness, lineType=4)
                    
                
                elif tool == tools[2]:
                    cv2.circle(image, (x_thumb, y_thumb), 2*thickness, brush_color, -1, lineType=4)
                    if  draw:
                        if ((x_end != x_start) or (y_end != y_start)):
                            circle.append(((x_start, y_start), (x_end, y_end), brush_color, 2*thickness, 4))
                            active.append(2)
                        x_start, y_start = x_index, y_index
                        x_end, y_end = x_index, y_index
                    if not draw:
                        x_end, y_end= x_index, y_index
                        if x_start is not None:
                            cv2.circle(image, (int((x_start + x_end)/2), int((y_start + y_end)/2)), int(dist(x_start, y_start, x_end, y_end)/2), brush_color, 2*thickness, lineType=4)
                
                elif tool == tools[3]:
                    cv2.circle(image, (x_thumb, y_thumb), 2*thickness, brush_color, -1, lineType=4)
                    if  draw:
                        if ((x_end != x_start) or (y_end != y_start)):
                            rectangle.append(((x_start, y_start), (x_end, y_end), brush_color, 2*thickness, 4))
                            active.append(3)
                        x_start, y_start = x_index, y_index
                        x_end, y_end = x_index, y_index
                    if not draw:
                        x_end, y_end= x_index, y_index
                        if x_start is not None:
                            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), brush_color, 2*thickness, lineType=4)

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

    if capture_canvas:
        cv2.imwrite('canvas_image.jpg', canvas)
        st.success('Canvas image captured!')

        with open("canvas_image.jpg", "rb") as file:
            st.download_button(label = "Download image", file_name="canvas.jpg", data= file)

        break
    
st.session_state.backup = [freehand , line, circle, rectangle, active, index]






