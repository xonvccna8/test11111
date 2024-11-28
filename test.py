import cv2
import socketio
import base64
import time

# Server URL
SERVER_URL = "http://localhost:5555"

# Initialize SocketIO client
sio = socketio.Client()

# Connect to the server
sio.connect(SERVER_URL)

# Event handlers
@sio.event
def connect():
    print("Connected to server")

@sio.event
def disconnect():
    print("Disconnected from server")

@sio.on('response')
def on_response(data):
    print("Received response:", data)

def send_frame(frame, room_participant_id):
    # Resize frame to reduce data size
    frame_resized = cv2.resize(frame, (320, 240))

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame_resized)

    # Base64 encode the image
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Send frame data
    data = {
        'roomParticipantId': room_participant_id,
        'image': img_base64
    }
    sio.emit('frame', data)

def main():
    # Setup camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Cannot open camera")
        return

    room_participant_id = "test_participant_001"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera")
            break

        # Send frame to server
        send_frame(frame, room_participant_id)

        # Display the frame locally
        cv2.imshow('Camera Feed', frame)

        # Limit frame rate to 5 FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()

if __name__ == "__main__":
    main()
