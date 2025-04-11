import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Advanced Image Processing App", layout="wide")

st.title("Image Processing App")
st.markdown("Upload an image and apply **cool filters** in real time!")

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

# --- Filter Functions ---
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

# --- Main App ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_column_width=True)

    with st.sidebar:
        st.header("🎨 Filters & Settings")
        option = st.radio("Choose a Filter:", [
            "Grayscale", "Canny Edge Detection", "Blur", "Sepia", "Invert Colors", "Sketch"
        ])

        settings = {}
        if option == "Canny Edge Detection":
            with st.expander("⚙️ Edge Detection Settings"):
                settings['low'] = st.slider("Min Threshold", 0, 100, 50)
                settings['high'] = st.slider("Max Threshold", 100, 300, 150)

        elif option == "Blur":
            with st.expander("⚙️ Blur Settings"):
                settings['k'] = st.slider("Kernel Size", 1, 15, 5, step=2)

    # Apply Selected Filter
    processed_img = None

    if option == "Grayscale":
        processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif option == "Canny Edge Detection":
        processed_img = cv2.Canny(img_array, settings['low'], settings['high'])
    elif option == "Blur":
        k = settings['k']
        processed_img = cv2.GaussianBlur(img_array, (k, k), 0)
    elif option == "Sepia":
        processed_img = apply_sepia(img_array)
    elif option == "Invert Colors":
        processed_img = cv2.bitwise_not(img_array)
    elif option == "Sketch":
        processed_img = apply_sketch(img_array)

    with col2:
        st.subheader(f"🧪 {option} Result")
        if len(processed_img.shape) == 2:
            st.image(processed_img, use_column_width=True, channels="GRAY")
        else:
            st.image(processed_img, use_column_width=True)

else:
    st.info("👈 Please upload an image from the sidebar to get started.")
