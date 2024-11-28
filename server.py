from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from modules.SCRFD import SCRFD
import base64
import eventlet
import requests
from flask_cors import CORS

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*")

# Load the SCRFD model
onnxmodel = 'models/scrfd_500m_kps.onnx'
mynet = SCRFD(onnxmodel)

# Dictionary to hold per-client data
client_data = {}

def find_pose(points):
    # Pose estimation code remains the same
    LMx = points[:,0]
    LMy = points[:,1]
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes)
    
    alpha = np.cos(angle)
    beta = np.sin(angle)
    
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    
    Xfrontal = (-90 + 90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90 + 90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return angle * 180 / np.pi, Xfrontal, Yfrontal

def report_violation(room_participant_id, violation_data):
    url = 'http://localhost:8000/api/violations'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'room_participant_id': room_participant_id,
        'left': violation_data['left'],
        'right': violation_data['right'],
        'up': violation_data['up'],
        'down': violation_data['down'],
        'multiple_people_count': violation_data['multiple_people_count']
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response
    except Exception as e:
        print(f"Error reporting violation: {e}")
        return None

@socketio.on('connect')
def handle_connect():
    # Initialize per-client data
    client_data[request.sid] = {
        'violation_counts': {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0,
            'multiple_people_count': 0
        },
        'violated': {
            'left': False,
            'right': False,
            'up': False,
            'down': False,
            'multiple_people_count': False,
        },
        'room_participant_id': None
    }
    print(f'Client {request.sid} connected')

@socketio.on('disconnect')
def handle_disconnect():
    # Remove client data on disconnect
    client_data.pop(request.sid, None)
    print(f'Client {request.sid} disconnected')

@socketio.on('frame')
def handle_frame(data):
    sid = request.sid
    if sid not in client_data:
        # Initialize per-client data if not present
        client_data[sid] = {
            'violation_counts': {
                'left': 0,
                'right': 0,
                'up': 0,
                'down': 0,
                'multiple_people_count': 0
            },
            'violated': {
                'left': False,
                'right': False,
                'up': False,
                'down': False,
                'multiple_people_count': False,
                'no_person': False
            },
            'no_person_count': 0,  # Add no person count
            'room_participant_id': None
        }

    # Access per-client data
    violation_counts = client_data[sid]['violation_counts']
    violated = client_data[sid]['violated']

    # Get room participant ID
    room_participant_id = data.get('roomParticipantId')
    if not room_participant_id:
        emit('error', {'message': 'Invalid roomParticipantId'})
        return
    else:
        client_data[sid]['room_participant_id'] = room_participant_id

    img_base64 = data.get('image')

    if not img_base64:
        emit('error', {'message': 'Invalid image data'})
        return

    # Decode base64 image
    img_bytes = base64.b64decode(img_base64)
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect faces
    bboxes, lmarks, scores = mynet.detect(frame)

    if bboxes.shape[0] > 0:  # If faces are detected
        violated['no_person'] = False  # Reset no person flag
        client_data[sid]['no_person_count'] = 0  # Reset counter

        # Multiple people detection
        if bboxes.shape[0] >= 2:
            if not violated['multiple_people_count']:
                violation_counts['multiple_people_count'] += 1
                violated['multiple_people_count'] = True
        else:
            violated['multiple_people_count'] = False

        # Pose estimation for the first detected face
        roll, yaw, pitch = find_pose(lmarks[0])

        # Left/Right violations
        if yaw > 40:
            if not violated['left']:
                violation_counts['left'] += 1
                violated['left'] = True
        elif yaw < -40:
            if not violated['right']:
                violation_counts['right'] += 1
                violated['right'] = True
        else:
            violated['left'] = False
            violated['right'] = False

        # Up/Down violations
        if pitch > 25:
            if not violated['up']:
                violation_counts['up'] += 1
                violated['up'] = True
        elif pitch < -25:
            if not violated['down']:
                violation_counts['down'] += 1
                violated['down'] = True
        else:
            violated['up'] = False
            violated['down'] = False
    else:  # No faces detected
        client_data[sid]['no_person_count'] += 1
        if not violated['no_person']:
            violated['no_person'] = True

    # Prepare violation data
    violation_data = {
        'left': violation_counts['left'],
        'right': violation_counts['right'],
        'up': violation_counts['up'],
        'down': violation_counts['down'],
        'multiple_people_count': violation_counts['multiple_people_count'],
        'no_person_count': client_data[sid]['no_person_count']  # Add no person count
    }

    # Report violations to backend
    report_violation(room_participant_id, violation_data)

    # Send response back to client
    response_data = {
        'roomParticipantId': room_participant_id,
        'violations': violation_data,
        'current_violations': violated
    }
    emit('response', response_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5555)
