import math
import cv2
import numpy as np
from ultralytics import YOLO
import torch
 
# Load the model
model = YOLO('yolov8x-pose-p6.pt')  # Load an official model
 
video_path = r'D:\Internship\Test\btrial9\istockphoto-1433501568-640_adpp_is.mp4'  # Input video file path
output_path = r'D:\Internship\Test\btrial9\output\istockphoto-1433501568-640_adpp_is.mp4'
 
# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator != 0:
        angle_rad = np.arccos(np.dot(v1, v2) / denominator)
    else:
        angle_rad = 0  # Set a default value if denominator is zero
    angle_deg = np.degrees(angle_rad)
    return angle_deg
 
# Function to determine if a person is standing
def is_standing(left_knee_angle, right_knee_angle, left_shoulder_hip_knee_angle, right_shoulder_hip_knee_angle):
    # Normalize angles if they exceed 180 degrees
    if left_shoulder_hip_knee_angle > 180:
        left_shoulder_hip_knee_angle = 360 - left_shoulder_hip_knee_angle
    if right_shoulder_hip_knee_angle > 180:
        right_shoulder_hip_knee_angle = 360 - right_shoulder_hip_knee_angle
   
    # Define thresholds for standing
    shoulder_hip_knee_threshold = 140  # Adjust as per your requirements
    knee_angle_threshold = 150  # Adjust as per your requirements
   
    # Check conditions for standing posture
    if (left_shoulder_hip_knee_angle > shoulder_hip_knee_threshold and left_knee_angle > knee_angle_threshold) or \
       (right_shoulder_hip_knee_angle > shoulder_hip_knee_threshold and right_knee_angle > knee_angle_threshold):
        return True
    return False
 
# Function to determine if a person is sitting
def is_sitting(left_body_angle, right_body_angle):
    # Define angle range for sitting
    body_angle_range = (95, 145)  # Adjust as per your requirements
   
    # Check if both body angles are within the defined range
    if body_angle_range[0] <= left_body_angle <= body_angle_range[1] and \
       body_angle_range[0] <= right_body_angle <= body_angle_range[1]:
        return True
    return False
   
# Function to determine if a person is sitting idle
def is_sitting_idle(sitting_start_frame, current_frame, fps):
    idle_threshold_frames = int(fps) * 2  # Idle threshold as 2 seconds
    idle = (current_frame - sitting_start_frame) > idle_threshold_frames
    print(f"Sitting Idle: {idle}, Start Frame: {sitting_start_frame}, Current Frame: {current_frame}, FPS: {fps}")
    return idle
 
def is_bending(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
    # Calculate angles
    left_shoulder_hip_knee_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_shoulder_hip_knee_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
 
    # Define thresholds for bending
    shoulder_hip_knee_threshold = 100  # Angle near 100 degrees for shoulder-hip-knee
    knee_angle_threshold = 165  # Angle near 165 degrees for knee
   
    # Check conditions for bending posture
    if (shoulder_hip_knee_threshold - 10 <= left_shoulder_hip_knee_angle <= shoulder_hip_knee_threshold + 10 or \
        shoulder_hip_knee_threshold - 10 <= right_shoulder_hip_knee_angle <= shoulder_hip_knee_threshold + 10) and \
       (knee_angle_threshold - 10 <= left_knee_angle <= knee_angle_threshold + 10 or \
        knee_angle_threshold - 10 <= right_knee_angle <= knee_angle_threshold + 10):
        return True
    return False
 
# Function to determine if a person is squatting
def is_squatting(left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle):
    if left_knee_angle > 180 or right_knee_angle > 180:
        left_knee_angle = 360 - left_knee_angle
        right_knee_angle = 360 - right_knee_angle
    knee_angle_range = (60, 100)  # Adjusted range for squatting detection
    hip_angle_range = (60, 100)
    knees_in_range = knee_angle_range[0] <= left_knee_angle <= knee_angle_range[1] and \
                     knee_angle_range[0] <= right_knee_angle <= knee_angle_range[1]
    hips_in_range = hip_angle_range[0] <= left_hip_angle <= hip_angle_range[1] and \
                    hip_angle_range[0] <= right_hip_angle <= hip_angle_range[1]
    squatting = knees_in_range and hips_in_range
    print(f"Squatting: {squatting}, Left Knee: {left_knee_angle}, Right Knee: {right_knee_angle}, Left Hip: {left_hip_angle}, Right Hip: {right_hip_angle}")
    return squatting
 
# Initialize video capture and writer
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
 
# Calculate frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)
 
# Dictionary to keep track of each person's sitting state and duration
person_sitting_state = {}
 
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    frame_count += 1
    print(f"Processing frame {frame_count}")
 
    # Predict with the model
    results = model(frame, conf=0.6, show_labels=True, show_conf=True, show=False, boxes=False)
 
    # Initialize counts
    standing_count = 0
    sitting_count = 0
    squatting_count = 0
    bending_count = 0
 
    # Extract keypoints from the results
    for person_id, result in enumerate(results):
        if hasattr(result, 'keypoints'):
            keypoints = result.keypoints.xy.cpu().numpy()  # Convert to numpy array
 
            if keypoints.shape[0] > 0 and keypoints.shape[1] >= 5:
                for person_idx in range(keypoints.shape[0]):
                    unique_id = person_id * keypoints.shape[0] + person_idx
                    if unique_id not in person_sitting_state:
                        person_sitting_state[unique_id] = {"sitting_start_frame": None}
 
                    nose = keypoints[person_idx][0][:2]
                    left_eye = keypoints[person_idx][1][:2]
                    right_eye = keypoints[person_idx][2][:2]
                    left_ear = keypoints[person_idx][3][:2]
                    right_ear = keypoints[person_idx][4][:2]
                    left_shoulder = keypoints[person_idx][5][:2]
                    right_shoulder = keypoints[person_idx][6][:2]
                    left_elbow = keypoints[person_idx][7][:2]
                    right_elbow = keypoints[person_idx][8][:2]
                    left_wrist = keypoints[person_idx][9][:2]
                    right_wrist = keypoints[person_idx][10][:2]
                    left_hip = keypoints[person_idx][11][:2]
                    right_hip = keypoints[person_idx][12][:2]
                    left_knee = keypoints[person_idx][13][:2]
                    right_knee = keypoints[person_idx][14][:2]
                    left_ankle = keypoints[person_idx][15][:2]
                    right_ankle = keypoints[person_idx][16][:2]
 
                    # Calculate the angles
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    left_body_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                    right_body_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
 
                    # Determine posture
                    posture = "none"
                    if is_standing(left_knee_angle, right_knee_angle, left_body_angle, right_body_angle):
                        standing_count += 1
                        posture = "standing"
                    elif is_sitting(left_body_angle, right_body_angle):
                        sitting_count += 1
                        posture = "sitting"
                        if person_sitting_state[unique_id]["sitting_start_frame"] is None:
                            person_sitting_state[unique_id]["sitting_start_frame"] = frame_count
                        elif is_sitting_idle(person_sitting_state[unique_id]["sitting_start_frame"], frame_count, fps):
                            posture = "sitting_idle"
                    elif is_squatting(left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle):
                        squatting_count += 1
                        posture = "squatting"
                    elif is_bending(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
                        bending_count += 1
                        posture = "bending"
 
                    print(f"Person ID: {unique_id}, Posture: {posture}")
 
                    # Bounding box around the person
                    min_x = int(min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
                    max_x = int(max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
                    min_y = int(min(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]))
                    max_y = int(max(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]))
 
                    # Draw the bounding box around the person
                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
 
                    # Annotate the frame with the posture label above the bounding box
                    cv2.putText(frame, posture, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)
 
                    # Annotate the frame with the angles
                    cv2.putText(frame, f"L: {left_knee_angle:.2f}",
                                (int(left_knee[0]), int(left_knee[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"R: {right_knee_angle:.2f}",
                                (int(right_knee[0]), int(right_knee[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2, cv2.LINE_AA)
 
                    # Annotate the frame with the person ID in red color
                    cv2.putText(frame, f"ID: {unique_id}", (min_x, min_y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2, cv2.LINE_AA)
                   
 
    # Annotate the frame with the counts
    cv2.rectangle(frame, (10, 10), (300, 140), (255, 255, 255), -1)  # White filled rectangle
    cv2.putText(frame, f"Standing: {standing_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Sitting: {sitting_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Squatting: {squatting_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Bending: {bending_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
 
    cv2.imshow("frame", cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    # Write the annotated frame to the output video
    out.write(frame)
 
cap.release()
out.release()
cv2.destroyAllWindows()
 
print(f"Annotated video saved at {output_path}")