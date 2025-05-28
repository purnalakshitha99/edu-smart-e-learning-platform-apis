import threading
import dlib
from flask import Blueprint, request, jsonify
from imutils import face_utils
from app.db import mongo
import time
import base64
import numpy as np
import cv2
from flask_cors import cross_origin
from app.models.drowsiness_detection import detect_drowsiness, detector, predictor, \
    eye_aspect_ratio, lip_distance, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, alarm, \
    YAWN_THRESH

# Initialize Flask Blueprint
warnings_bp = Blueprint('warnings', __name__)

COUNTER = 0
alarm_status = False
alarm_status2 = False

@warnings_bp.route('/process_frame', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')
def process_frame():
    global COUNTER, alarm_status, alarm_status2
    if request.method == 'OPTIONS':
        return '', 204

    # Get JSON data from request
    data = request.get_json()
    frame_data = data.get('frame')

    if not frame_data:
        return jsonify({"error": "No frame data provided"}), 400

    try:
        # Decode the frame from base64
        frame = decode_frame(frame_data)
    except Exception as e:
        return jsonify({"error": f"Failed to decode frame: {str(e)}"}), 400

    # Call the detect_drowsiness function
    processed_frame, warning_message, sleep_detected, yawn_detected = detect_drowsiness(frame)

    # Convert processed frame to base64
    _, img_encoded = cv2.imencode('.jpg', processed_frame)
    processed_frame_bytes = img_encoded.tobytes()

    # Return both the processed frame and warning flags
    return jsonify({
        "processed_frame": base64.b64encode(processed_frame_bytes).decode('utf-8'),
        "warnings": [warning_message],
        "alarm_triggered": sleep_detected,  # Only trigger alarm sound for sleep detection
        "notification_triggered": sleep_detected or yawn_detected  # Notification for both
    })

# Endpoint to get warnings for a specific student
@warnings_bp.route('/get_warnings/<student_id>', methods=['GET'])
@cross_origin()
def get_warnings(student_id):
    warnings_collection = mongo.db.warnings
    student_warnings = warnings_collection.find({"student_id": student_id})
    warning_messages = [warning["warning_message"] for warning in student_warnings]
    return jsonify({"warnings": warning_messages})

def decode_frame(base64_data):
    # Decode the base64 data
    nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)

    # Decode into an image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("The decoded frame is empty. Please check the frame data.")

    return frame