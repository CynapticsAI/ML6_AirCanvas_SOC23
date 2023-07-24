import streamlit as st
from PIL import Image


st.set_page_config(
    page_title = "USER MANUAL",

)
st.title("HOW TO USE AIR CANVAS 🤔?")
st.sidebar.success("select a page above.")


st.subheader("Tick the checkbox 'RUN' to start using the air canvas.")
image = Image.open('final_air_canvas/run.png')
st.image(image,  channels="BGR" )

st.subheader("Use your index finger to draw , join index finger and thumb to stop drawing from the pen.")
draw = Image.open('final_air_canvas/draw.png')
st.image(draw,  channels="BGR" )

st.subheader("Use the clear button to clear the canvas.")
clear = Image.open('final_air_canvas/clear.png')
st.image(clear,  channels="BGR" )

st.subheader("You can change the color of brush, color of canvas, adjust the thickness of brush from the sidebar.")
color = Image.open('final_air_canvas/color.png')
st.image(color,  channels="BGR" )

st.subheader("From the drawing tool in the sidebar, select the tool you wish to use.")
drawing_tool = Image.open('final_air_canvas/drawing_tool.png')
st.image(drawing_tool,  channels="BGR" )

st.subheader("Join index finger and thumb to start drawing shapes, then stretch the shape as per your required size then separate your index finger and thumb to get the required shape.")
shapes = Image.open('final_air_canvas/shapes.png')
st.image(shapes,  channels="BGR" )

st.subheader("Click on the UNDO button in the sidebar to undo the previous operation.")
undo = Image.open('final_air_canvas/undo.png')
st.image(undo,  channels="BGR" )

st.subheader("Click on the 'CAPTURE CANVAS'button in the sidebar to capture the canvas, then click on the 'DOWNLOAD IMAGE' to download the image.")
cap_button = Image.open('final_air_canvas/cap_button.png')
st.image(cap_button,  channels="BGR" )
dow = Image.open('final_air_canvas/dow.png')
st.image(dow,  channels="BGR" )

