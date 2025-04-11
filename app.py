import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Advanced Image Processing App")
st.write("Upload an image and apply various filters.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def apply_sepia(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def apply_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.sidebar.title("Choose a Filter")
    option = st.sidebar.radio(
        "Select a filter",
        ["Grayscale", "Canny Edge Detection", "Blur", "Sepia", "Invert Colors", "Sketch"]
    )

    if option == "Grayscale":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.subheader("Grayscale Image")
        st.image(gray, use_column_width=True, channels="GRAY")

    elif option == "Canny Edge Detection":
        low = st.sidebar.slider("Min Threshold", 0, 100, 50)
        high = st.sidebar.slider("Max Threshold", 100, 300, 150)
        edges = cv2.Canny(img_array, low, high)
        st.subheader("Edge Detected Image")
        st.image(edges, use_column_width=True, channels="GRAY")

    elif option == "Blur":
        k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
        blurred = cv2.GaussianBlur(img_array, (k, k), 0)
        st.subheader("Blurred Image")
        st.image(blurred, use_column_width=True)

    elif option == "Sepia":
        sepia_img = apply_sepia(img_array)
        st.subheader("Sepia Image")
        st.image(sepia_img, use_column_width=True)

    elif option == "Invert Colors":
        inverted = cv2.bitwise_not(img_array)
        st.subheader("Inverted Color Image")
        st.image(inverted, use_column_width=True)

    elif option == "Sketch":
        sketch_img = apply_sketch(img_array)
        st.subheader("Sketch Effect")
        st.image(sketch_img, use_column_width=True, channels="GRAY")
