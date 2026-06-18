import streamlit as st
import tempfile
import os
from pathlib import Path

# Import the detector module from the existing package
from speed_detector import detect_vehicles

st.set_page_config(page_title="Vehicle Speed Detector", layout="wide")
st.title("Vehicle Speed Detector")

st.sidebar.header("Detection settings")
model = st.sidebar.text_input("YOLO model path", value="yolov8n.pt")
distance_meters = st.sidebar.number_input("Distance between lines A and B (meters)", value=10.0)
speed_limit_kmh = st.sidebar.number_input("Speed limit (km/h, 0 to disable)", value=60.0)
max_speed_kmh = st.sidebar.number_input("Max speed filter (km/h, 0 to disable)", value=150.0)
confidence = st.sidebar.slider("Detection confidence", 0.0, 1.0, 0.25)
detect_roi_only = st.sidebar.checkbox("Detect ROI only (mask outside trapezoid)", value=False)

# Default line specs come from the original module
default_line_a = detect_vehicles.DEFAULT_CONFIG["line_a"]
default_line_b = detect_vehicles.DEFAULT_CONFIG["line_b"]
line_a = st.sidebar.text_input("Line A (x1,y1,x2,y2 ratios)", value=default_line_a)
line_b = st.sidebar.text_input("Line B (x1,y1,x2,y2 ratios)", value=default_line_b)

st.write("Upload a video file (mp4, avi, mov, mkv). The app will run the detector and show the annotated video and a CSV of speeds.")
uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"]) 

run = st.button("Run detection")

if uploaded is not None and run:
    tmpdir = tempfile.mkdtemp(prefix="speeddet_")
    input_path = os.path.join(tmpdir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    output_path = os.path.join(tmpdir, "detected.mp4")
    csv_path = os.path.join(tmpdir, "speed_results.csv")

    st.info("Running detection. This may take time and requires the YOLO model and dependencies.")
    with st.spinner("Processing..."):
        try:
            detect_vehicles.run_detection(
                source=input_path,
                model_path=model,
                confidence=confidence,
                tracker=None,
                output_path=output_path,
                output_dir="",
                show_window=False,
                max_frames=0,
                line_a_spec=detect_vehicles.parse_line_spec(line_a),
                line_b_spec=detect_vehicles.parse_line_spec(line_b),
                detect_roi_only=detect_roi_only,
                distance_meters=distance_meters,
                max_speed_kmh=max_speed_kmh,
                speed_limit_kmh=speed_limit_kmh,
                csv_path=csv_path,
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        st.video(output_path)
    else:
        st.warning("No output video produced.")

    if os.path.exists(csv_path):
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            st.subheader("Speed records")
            st.dataframe(df)
        except Exception:
            st.write("Saved CSV at:", csv_path)

    st.success(f"Done. Temporary files are in: {tmpdir}")
    st.write("Tip: Download files from the temporary directory if you need to keep them.")
else:
    st.info("Upload a video and press 'Run detection' to start.")
