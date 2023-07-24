import streamlit as st
from PIL import Image


st.set_page_config(
    page_title = "USER MANUAL",
)

st.title("HOW TO USE AIR CANVAS 🤔?")
st.sidebar.success("select a page above.")
st.text("")

st.subheader("Tick the checkbox 'RUN' to start using the air canvas.")
image = Image.open('pages/run.png')
st.image(image,  channels="BGR" )
st.text("")

st.subheader("Use your index finger to draw , join index finger and thumb to stop drawing from the pen.")
draw = Image.open('pages/draw.png')
st.image(draw,  channels="BGR" )
st.text("")

st.subheader("Use the clear button to clear the canvas.")
clear = Image.open('pages/clear.png')
st.image(clear,  channels="BGR" )
st.text("")

st.subheader("You can change the color of brush, color of canvas, adjust the thickness of brush from the sidebar.")
color = Image.open('pages/color.png')
st.image(color,  channels="BGR" )
st.text("")

st.subheader("From the drawing tool in the sidebar, select the tool you wish to use.")
drawing_tool = Image.open('pages/drawing_tool.png')
st.image(drawing_tool,  channels="BGR" )
st.text("")

st.subheader("Join index finger and thumb to start drawing shapes, then stretch the shape as per your required size then separate your index finger and thumb to get the required shape.")
shapes = Image.open('pages/shapes.png')
st.image(shapes,  channels="BGR" )
st.text("")

st.subheader("Click on the UNDO button in the sidebar to undo the previous operation.")
undo = Image.open('pages/undo.png')
st.image(undo,  channels="BGR" )
st.text("")

st.subheader("Click on the 'CAPTURE CANVAS'button in the sidebar to capture the canvas, then click on the 'DOWNLOAD IMAGE' to download the image.")
cap_button = Image.open('pages/cap_button.png')
st.image(cap_button,  channels="BGR" )
dow = Image.open('pages/dow.png')
st.image(dow,  channels="BGR" )

