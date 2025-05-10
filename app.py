from flask import Flask, render_template, redirect, url_for, session, request, flash, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
from ultralytics import YOLO
import threading
import time


app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your-secret-key-here'  # Change for production!

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your custom trained model

# Dummy users
users = {
    "admin": generate_password_hash("tnmc@123"),
    "inspector": generate_password_hash("tnmc@456")
}

# Global variables for frame and detection results
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()

def run_detection():
    global latest_frame, latest_detections
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        results = model(frame, stream=True)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf > 0.5:  # Confidence threshold
                    detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        with frame_lock:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                latest_frame = jpeg.tobytes()
            latest_detections = detections
        
        time.sleep(0.03)
    
    cap.release()

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            frame = latest_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        flash("Invalid credentials", "error")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
@app.route('/index.html')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', active_page='index')

@app.route('/dashboard.html')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    if 'username' not in session:
        return redirect(url_for('login'))
    with frame_lock:
        return jsonify(latest_detections)

if __name__ == '__main__':
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
