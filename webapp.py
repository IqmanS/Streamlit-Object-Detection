import streamlit as st
from PIL import Image
import torch
import os
import urllib.request

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.title('Harmful Object Detection')

img = st.selectbox(options=("Upload New Image",'img1.jpg', 'img2.jpg','img3.jpg'),label="Select a Sample Image or Link a New Image")

# model_name = st.sidebar.selectbox(
#     'Select Model',
#     ('YOLOv5s', 'FasterRCNN-MobileNetV3', 'FasterRCNN-ResNet50',)
# )

model = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo-Wts/best.pt', force_reload=True)

image = "test-images/img1"
input_image = ""
img_loaded = False
if img == "Upload New Image":
    url = st.text_input(label="Image URL")
    if len(url) >10:
        try:
            img = "img-uploaded.jpg"
            urllib.request.urlretrieve(url, "test-images/"+img)
            st.write('### Source image:')
            input_image = "test-images/" + img
            image = Image.open(input_image)
            st.image(image, width=400)  # image: numpy array
            img_loaded= True
        except:
            st.write('Invalid URL')

else:
    st.write('### Source image:')
    input_image = "test-images/"+img
    image = Image.open(input_image)
    st.image(image, width=400)  # image: numpy array
    img_loaded = True

# clickedLoad = st.button('Load Image')
# if clickedLoad:
clickedDetect=False
if img_loaded:
    clickedDetect = st.button('Detect')

if clickedDetect:
    op = model(input_image)
    op.save()
    fol = os.listdir("runs/detect/")[-1]
    output_image = "runs/detect/"+fol +"/" + img
    st.write('### Output image:')
    op_image = Image.open(output_image)
    st.image(op_image, width=400)




