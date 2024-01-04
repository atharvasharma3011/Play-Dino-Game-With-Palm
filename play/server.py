from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

detector = HandDetector(detectionCon=0.8, maxHands=1)
hand_gesture = "No Gesture"  # Variable to store hand gesture


def detect_hand(frame):
    hands, _ = detector.findHands(frame)

    if hands:
        lmList = hands[0]
        fingers_up = detector.fingersUp(lmList)

        if fingers_up == [0, 0, 0, 0, 0]:
            hand_gesture = "Fist"
        elif fingers_up == [1, 1, 1, 1, 1]:
            hand_gesture = "Open Hand"
        # Add more conditions based on your gestures

    else:
        hand_gesture = "No Gesture"

    return hand_gesture


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        hand_gesture = detect_hand(frame)

        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
