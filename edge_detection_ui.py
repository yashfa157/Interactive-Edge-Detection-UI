# edge_detection_app.py
# Streamlit app for interactive edge detection experiments
# Requirements: streamlit, opencv-python, pillow, numpy

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Edge Detection Interface", layout="wide")

st.title("Edge Detection Interface")
st.write("Upload an image and experiment with Sobel, Laplacian, and Canny edge detectors.")

# Sidebar: upload and algorithm selection
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])
    algo = st.selectbox("Choose algorithm", ["Sobel", "Laplacian", "Canny"])
    st.markdown("---")

    # Algorithm specific parameters
    if algo == "Canny":
        st.subheader("Canny parameters")
        lower = st.slider("Lower threshold", 0, 500, 50)
        upper = st.slider("Upper threshold", 0, 500, 150)
        ksize = st.selectbox("Gaussian kernel size", [1, 3, 5, 7, 9], index=1)
        sigma = st.slider("Gaussian sigma", 0.0, 10.0, 1.0, step=0.1)
        use_blur = st.checkbox("Apply Gaussian blur before Canny", value=True)
    elif algo == "Sobel":
        st.subheader("Sobel parameters")
        ksize = st.selectbox("Kernel size", [1, 3, 5, 7], index=1)
        grad_dir = st.radio("Gradient direction", ["X", "Y", "Both"], index=2)
    else:  # Laplacian
        st.subheader("Laplacian parameters")
        ksize = st.selectbox("Kernel size", [1, 3, 5, 7], index=1)

    st.markdown("---")
    realtime = st.checkbox("Realtime update", value=True)
    if not realtime:
        apply_btn = st.button("Apply")
    else:
        apply_btn = True


# Utility functions
def load_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to BGR numpy array for OpenCV."""
    img = np.array(pil_image.convert("RGB"))
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    return img


def to_displayable(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR or single channel image to PIL for st.image."""
    if img is None:
        return None
    if len(img.shape) == 2:
        # single channel to RGB for display
        return Image.fromarray(img)
    else:
        # BGR to RGB
        rgb = img[:, :, ::-1]
        return Image.fromarray(rgb)


def apply_canny(img_bgr, lower_t, upper_t, ksize_blur=3, sigma=1.0, do_blur=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if do_blur and ksize_blur > 1:
        # OpenCV requires odd kernel
        k = ksize_blur if ksize_blur % 2 == 1 else ksize_blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), sigma)
    edges = cv2.Canny(gray, lower_t, upper_t)
    return edges


def apply_sobel(img_bgr, ksize=3, direction="Both"):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = ksize if ksize % 2 == 1 else ksize + 1
    # dx, dy
    if direction == "X":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        abs_sx = cv2.convertScaleAbs(sx)
        return abs_sx
    elif direction == "Y":
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        abs_sy = cv2.convertScaleAbs(sy)
        return abs_sy
    else:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(sx, sy)
        mag = cv2.convertScaleAbs(mag)
        return mag


def apply_laplacian(img_bgr, ksize=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = ksize if ksize % 2 == 1 else ksize + 1
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    lap_abs = cv2.convertScaleAbs(lap)
    return lap_abs


# Main UI area - side by side displays
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Input - Original Image")
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, use_container_width=True)
    else:
        st.info("Please upload an image from the sidebar to begin.")

with right_col:
    st.subheader("Output - Edge Detection Result")
    if uploaded_file is not None and apply_btn:
        pil_img = Image.open(uploaded_file)
        img_bgr = load_image(pil_img)

        if algo == "Canny":
            edges = apply_canny(img_bgr, lower, upper, ksize_blur=ksize, sigma=sigma, do_blur=use_blur)
            disp = to_displayable(edges)
            st.caption(f"Canny - lower={lower}, upper={upper}, blur_k={ksize}, sigma={sigma}")
            st.image(disp, use_container_width=True)
        elif algo == "Sobel":
            edges = apply_sobel(img_bgr, ksize=ksize, direction=grad_dir)
            disp = to_displayable(edges)
            st.caption(f"Sobel - ksize={ksize}, direction={grad_dir}")
            st.image(disp, use_container_width=True)
        else:  # Laplacian
            edges = apply_laplacian(img_bgr, ksize=ksize)
            disp = to_displayable(edges)
            st.caption(f"Laplacian - ksize={ksize}")
            st.image(disp, use_container_width=True)
    else:
        st.info("No output to show. Upload an image and press Apply if realtime update is disabled.")

# Footer
st.markdown("---")
st.write("Tips: Try increasing Canny thresholds to reduce edges. Use Sobel X or Y to see directional gradients. Laplacian highlights second order changes.")

# End of file
