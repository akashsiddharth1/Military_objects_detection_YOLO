import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
model = YOLO('best.pt')  # Replace with your trained YOLO model path

# Helper to convert detection results to threat / non-threat
def classify_threat(classes):
    threat_classes = ['weapon', 'enemy_soldier', 'military_tank', 'military_truck']
    if any(cls in classes for cls in threat_classes):
        return 'Threat Detected'
    else:
        return 'Non-Threat'

# Streamlit App UI
st.title("Weapon & Threat Detection System")
st.sidebar.header("Upload Media")

upload_option = st.sidebar.radio("Choose Input Type", ["Image", "Video"])

if upload_option == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Original Image', use_column_width=True)

        img_array = np.array(img.convert('RGB'))

        # YOLO Inference
        results = model.predict(img_array, conf=0.3)
        res_img = results[0].plot()

        # Extract detected classes
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

        # Display
        st.image(res_img, caption='Detection Result', use_column_width=True)
        st.write(f"**Classification:** {classify_threat(detected_classes)}")
        st.write(f"**Detected Classes:** {detected_classes}")

        # Download button
        res_pil = Image.fromarray(res_img)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        res_pil.save(tmp_file.name)
        st.download_button(label="Download Annotated Image", data=open(tmp_file.name, 'rb').read(), file_name='result.png', mime='image/png')

if upload_option == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=0.3)
            res_frame = results[0].plot()

            if out is None:
                h, w = res_frame.shape[:2]
                out = cv2.VideoWriter(out_file.name, fourcc, 20.0, (w, h))

            out.write(res_frame)
            stframe.image(res_frame, channels="BGR")

        cap.release()
        out.release()

        st.success("Video processing completed.")
        st.download_button("Download Processed Video", open(out_file.name, 'rb').read(), file_name='result.mp4', mime='video/mp4')
