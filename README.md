# AIR-CANVAS

Air-Canvas is an interactive drawing application that allows you to draw and create digital art using hand gestures captured by your webcam. The application uses computer vision and hand tracking to interpret your finger movements and translates them into brush strokes on a virtual canvas.

![Air-Canvas Demo](demo.gif)

## Features

- *Real-time Drawing*: Draw directly on the canvas using your finger movements.
- *Multiple Drawing Tools*: Choose from various drawing tools like Freehand, Line, Circle, and Rectangle.
- *Adjustable Brush Settings*: Customize the brush color and thickness to match your artistic vision.
- *Undo and Clear Canvas*: Easily undo the last action or clear the entire canvas to start anew.
- *Capture Canvas Image*: Save your artwork by capturing the current state of the canvas as an image.
- *User-Friendly Interface*: The intuitive interface makes drawing and interacting with the application a breeze.

## How to Use

1. Make sure your webcam is connected and functioning correctly.
2. Run the application and select a drawing tool from the sidebar.
3. Adjust the brush color and thickness to your liking using the color picker and slider in the sidebar.
4. Interact with the canvas using your index finger and thumb gestures:
   - For the Freehand tool, simply draw by moving your index finger while holding your thumb close together.
   - For other tools (Line, Circle, Rectangle), maintain a certain distance between your index finger and thumb to start drawing the shape.
5. Use the "Undo" button in the sidebar to remove the last drawing action if needed.
6. To clear the entire canvas, click the "CLEAR" button in the top left corner of the canvas area.
7. If you're satisfied with your artwork, use the "CAPTURE CANVAS" button in the sidebar to save the canvas image.

## Requirements

- Python 3.x
- Streamlit
- OpenCV
- Mediapipe
- NumPy
- Pillow

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
   pip install streamlit opencv-python mediapipe numpy pillow
3. For Using:
   streamlit run Air_Canvas.py
