import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Streamlit setup
st.title('BADMINTON BODY AND HAND SPEED TRACKER')
st.sidebar.title('Settings')
source_option = st.sidebar.selectbox('Select Video Source', ('Upload Video','Live Camera'))

# Initialize list to store points for drawing the movement line for the person
points_3d = []

# Variables to calculate speed
previous_position = None
speed_text = "Speed: 0.00 km/h"
speeds_kmh = []


# Define function to draw lines
def draw_lines(frame, points, color):
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2)


# Check the source option and initialize video capture accordingly
if source_option == 'Live Camera':
    camera_index = st.sidebar.slider('Select Camera Index', 0, 1, 1)
    cap = cv2.VideoCapture(camera_index)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file.name)
    else:
        cap = None

# Check if the camera or video opened successfully
if cap is None or not cap.isOpened():
    st.error("Error: Could not open camera or video.")
else:
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define a conversion factor from video units to meters
    units_to_meters = 0.02  # Example: 1 unit in video = 1 meter

    # Stream video frames
    frame_window = st.image([], use_column_width=True)
    plot_placeholder = st.empty()
    hist_placeholder = st.empty()

    # Set the frame skipping rate and resize scale
    frame_skip = st.sidebar.slider('Frame Skip', 1, 10, 7)
    resize_scale = st.sidebar.slider('Resize Scale', 0.1, 1.0, 0.5)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize the frame to speed up processing
        frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        result = pose.process(rgb_frame)

        # Check if any landmarks are detected
        if result.pose_landmarks:
            # Extract landmarks
            landmarks = result.pose_landmarks.landmark

            # Assign color for the person
            color = (0, 255, 0)

            # Draw the pose landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Track the right wrist
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            wrist_x = right_wrist.x * w
            wrist_y = right_wrist.y * h
            wrist_z = right_wrist.z * w  # assuming z is scaled similar to x and y

            # Calculate speed
            current_position = np.array([right_wrist.x, right_wrist.y, wrist_z])
            if previous_position is not None:
                distance_units = np.linalg.norm(current_position - previous_position)
                distance_meters = distance_units * units_to_meters
                speed_mps = distance_meters * fps / frame_skip  # Adjust speed calculation for frame skipping
                speed_kmh = speed_mps * 3.6
                speeds_kmh.append(speed_kmh)
                speed_text = f"Speed: {speed_kmh:.2f} km/h"
            previous_position = current_position

            # Append the points to the list
            points_3d.append((right_wrist.x, right_wrist.y, wrist_z))

            # Draw the line
            draw_lines(frame, [(int(p[0] * w), int(p[1] * h)) for p in points_3d], color)

            # Draw a circle on the current position
            cv2.circle(frame, (int(wrist_x), int(wrist_y)), 5, color, -1)

            # Display the speed on the frame
            cv2.putText(frame, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert frame to RGB for display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

        # Update 3D trajectory plot
        if points_3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xs, ys, zs = zip(*points_3d)
            ax.plot(xs, ys, zs, label='Person - Right Wrist', color='green')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.legend()
            plot_placeholder.pyplot(fig)

        # Update speed histogram
        if speeds_kmh:
            plt.figure()
            plt.hist(speeds_kmh, bins=20, color='blue', edgecolor='black')
            plt.title('Histogram of Hand Speed (km/h)')
            plt.xlabel('Speed (km/h)')
            plt.ylabel('Frequency')
            hist_placeholder.pyplot(plt)

    # Release resources
    cap.release()
