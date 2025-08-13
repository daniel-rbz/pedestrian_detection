from flask import Flask, Response, render_template, jsonify
import sqlite3
import threading
import time
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs.db')
latest_frames = {}  # This should be set by the main app
frames_lock = threading.Lock()

def get_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT * FROM logs ORDER BY id DESC LIMIT ?', (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.route('/')
def dashboard():
    with frames_lock:
        camera_ids = sorted(list(latest_frames.keys()))
    return render_template('dashboard.html', camera_ids=camera_ids)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    def gen():
        while True:
            with frames_lock:
                frame = latest_frames.get(camera_id)
            if frame is not None:
                import cv2
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    try:
        logs = get_logs()
        return jsonify({'logs': logs})
    except Exception as e:
        print("Error in /logs route:", e)
        return jsonify({'logs': [], 'error': str(e)}), 500

# The main app should import dashboard.py and start the Flask app in a background thread, passing latest_frames.

if __name__ == '__main__':
    app.run(debug=True, port=5000) 