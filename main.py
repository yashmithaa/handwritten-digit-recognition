import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

st.title("Handwritten Digit Recognition")
st.markdown("Draw here")

model = torch.load("recog-30.pt")
model.eval()

col1, col2 = st.columns(2)

with col1:
    canvas_result = st_canvas(
        fill_color = "#ffffff",
        stroke_width = 10,
        stroke_color = "#ffffff",
        background_color = "#000000",
        height = 150, width =150,
        drawing_mode = "freedraw",
        key = "canvas",
           
    )

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    with col2:
        st.image(img,caption ="Input")#,use_column_width = True)

u='''with col1:
    c1,c2,c3=st.columns(3)
    with col1:'''
b = st.button("Predict",type="primary",help="predicts the output",use_container_width = True)
if b:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        img = img[:,:,:3]
        img = Image.fromarray(img)
        img = img.convert("L")
        img = img.resize((28,28))
        img = transform(img).view(1,784)
        

        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        s = "predicted digit is"+str(probab.index(max(probab)))
        st.write(s)
        st.bar_chart(probab)