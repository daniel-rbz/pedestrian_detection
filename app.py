from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue, Empty
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import sqlite3
from dashboard import app as dashboard_app, latest_frames, frames_lock
import threading as th

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def setup_camera_logger(camera_id):
    """Setup separate logger for each camera"""
    logger = logging.getLogger(f"Camera_{camera_id}")
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, f'camera_{camera_id}_detection.log'))
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text):
        try:
            data = {"chat_id": self.chat_id, "text": text}
            requests.post(self.api_url, data=data, timeout=5)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

class NotificationThread:
    def __init__(self, telegram_notifier):
        self.telegram_notifier = telegram_notifier
        self.notification_queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        while self.running:
            try:
                message = self.notification_queue.get(timeout=1)
                self.telegram_notifier.send_message(message)
            except Empty:
                continue

    def send_notification(self, message):
        self.notification_queue.put(message)

class DatabaseLogger:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs.db')
        self.db_path = db_path
        self._setup_db()

    def _setup_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                event_type TEXT,
                person_id INTEGER,
                timestamp TEXT,
                duration REAL,
                total_people INTEGER
            )
        ''')
        self.conn.commit()

    def log_event(self, camera_id, event_type, person_id, timestamp, duration, total_people):
        self.cursor.execute('''
            INSERT INTO logs (camera_id, event_type, person_id, timestamp, duration, total_people)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (camera_id, event_type, person_id, timestamp, duration, total_people))
        self.conn.commit()

    def close(self):
        self.conn.close()

class PersonTracker:
    def __init__(self, camera_id, exit_buffer_seconds=3, notification_thread=None, db_logger=None):
        self.camera_id = camera_id
        self.tracked_people = {}  # track_id -> entry_time
        self.pending_exits = {}  # track_id -> (exit_time, entry_time)
        self.exit_buffer_seconds = exit_buffer_seconds
        self.db_logger = db_logger or DatabaseLogger()
        self.notification_thread = notification_thread

    def update_tracking(self, results):
        """Update tracking and log entry/exit events with buffer for stability"""
        if not results or len(results) == 0:
            return
        
        result = results[0]
        
        if not hasattr(result, 'boxes') or result.boxes is None:
            return
        
        current_track_ids = set()
        
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            current_track_ids = set(result.boxes.id.cpu().numpy().astype(int))
        
        current_time = datetime.now()
        
        # Check for people who have been pending exit long enough to confirm
        confirmed_exits = []
        for track_id, (exit_time, entry_time) in list(self.pending_exits.items()):
            if (current_time - exit_time).total_seconds() >= self.exit_buffer_seconds:
                confirmed_exits.append(track_id)
                duration = (exit_time - entry_time).total_seconds()
                total_people = len(self.tracked_people) + len(self.pending_exits) - 1
                msg = f"Camera {self.camera_id}: Person {track_id} LEFT frame after {duration:.1f}s. Total people: {total_people}"
                # Debug print
                print(f"track_id value: {track_id}, type: {type(track_id)} (LEFT event)")
                # Log to database, force int
                self.db_logger.log_event(
                    camera_id=self.camera_id,
                    event_type="LEFT",
                    person_id=int(track_id),
                    timestamp=exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    duration=duration,
                    total_people=total_people
                )
                if self.notification_thread:
                    self.notification_thread.send_notification(msg)
        
        for track_id in confirmed_exits:
            self.pending_exits.pop(track_id)
        
        # Find new people (entered) - check if they were in pending exits
        new_people = current_track_ids - set(self.tracked_people.keys()) - set(self.pending_exits.keys())
        for track_id in new_people:
            self.tracked_people[track_id] = current_time
            total_people = len(self.tracked_people) + len(self.pending_exits)
            msg = f"Camera {self.camera_id}: Person {track_id} ENTERED frame. Total people: {total_people}"
            # Debug print
            print(f"track_id value: {track_id}, type: {type(track_id)} (ENTERED event)")
            # Log to database, force int
            self.db_logger.log_event(
                camera_id=self.camera_id,
                event_type="ENTERED",
                person_id=int(track_id),
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                duration=None,
                total_people=total_people
            )
            if self.notification_thread:
                self.notification_thread.send_notification(msg)
        
        # Find people who disappeared (but might come back)
        disappeared_people = set(self.tracked_people.keys()) - current_track_ids
        for track_id in disappeared_people:
            entry_time = self.tracked_people.pop(track_id)
            self.pending_exits[track_id] = (current_time, entry_time)
        
        # Restore people who reappeared from pending exits
        reappeared_people = set(self.pending_exits.keys()) & current_track_ids
        for track_id in reappeared_people:
            exit_time, entry_time = self.pending_exits.pop(track_id)
            self.tracked_people[track_id] = entry_time

class CameraThread:
    def __init__(self, camera_id, input_source, model_path, output_queue, notification_thread=None, db_logger=None):
        self.camera_id = camera_id
        self.input_source = input_source  # Can be camera ID (int) or video path (str)
        self.model_path = model_path
        self.output_queue = output_queue
        self.running = False
        self.thread = None
        self.cap = None
        self.model = None
        self.person_tracker = PersonTracker(camera_id, notification_thread=notification_thread, db_logger=db_logger)

    def start(self):
        self.running = True
        if isinstance(self.input_source, int):
            self.cap = cv2.VideoCapture(self.input_source)
        else:
            self.cap = cv2.VideoCapture(self.input_source)
        self.model = YOLO(self.model_path)
        self.model.to("cuda")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.input_source, str):
                    # Video ended, restart from beginning for continuous simulation
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print(f"Camera {self.camera_id}: Failed to read frame")
                    continue

            start_time = time.time()
            results = self.model.track(source=frame, show=False, classes=[0], persist=True)
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            self.person_tracker.update_tracking(results)

            if results and len(results) > 0 and hasattr(results[0], 'plot'):
                frame = results[0].plot()
            
            current_people = len(self.person_tracker.tracked_people) + len(self.person_tracker.pending_exits)
            status_text = f"People: {current_people}" if current_people > 0 else "No People"
            
            source_type = "VIDEO" if isinstance(self.input_source, str) else "CAMERA"
            text = f"Camera {self.camera_id} ({source_type}) - {status_text} - FPS: {fps:.1f}"
            
            frame_height, frame_width = frame.shape[:2]
            
            font_scale = 0.6
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            max_text_width = frame_width * 0.9
            if text_width > max_text_width:
                font_scale = font_scale * (max_text_width / text_width) * 0.9
                font_thickness = max(1, int(font_thickness * 0.8))
            
            font_scale = max(0.3, font_scale)
            
            text_x = 10
            text_y = min(30, frame_height - 10)
            
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            
            self.output_queue.put((self.camera_id, frame))

def main():
    INPUT_MODE = "CAMERA"  # Options: "CAMERA" or "VIDEO"
    
    if INPUT_MODE == "CAMERA":
        NUM_CAMERAS = 1
        input_sources = list(range(NUM_CAMERAS))
    elif INPUT_MODE == "VIDEO":
        input_sources = [
            "Videos/Videos/sneak/YOUTUBE_YouTubeCCTV046_sneak_5.mp4",
            "Videos/Videos/sneak/UCFCRIME_Burglary064_sneak_1.mp4",
            "Videos/Videos/sneak/UCFCRIME_Burglary071_sneak_2.mp4",
        ]
    else:
        print("Invalid INPUT_MODE. Use 'CAMERA' or 'VIDEO'")
        return
    
    model_path = "yolo11n.pt"
    
    output_queue = Queue()
    
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    notification_thread = NotificationThread(telegram_notifier)
    
    camera_threads = []
    
    db_logger = DatabaseLogger()
    
    for camera_id, input_source in enumerate(input_sources):
        if INPUT_MODE == "VIDEO" and not os.path.exists(input_source):
            print(f"Warning: Video file not found for Camera {camera_id}: {input_source}")
            continue
            
        thread = CameraThread(camera_id, input_source, model_path, output_queue, notification_thread=notification_thread, db_logger=db_logger)
        thread.start()
        camera_threads.append(thread)
        
        if INPUT_MODE == "CAMERA":
            print(f"Started Camera {camera_id} (Real Camera)")
        else:
            print(f"Started Camera {camera_id} (Video: {input_source})")

    # Start Flask dashboard in a background thread
    flask_thread = th.Thread(target=lambda: dashboard_app.run(debug=False, port=5000, use_reloader=False), daemon=True)
    flask_thread.start()

    try:
        while True:
            while not output_queue.empty():
                camera_id, frame = output_queue.get()
                # Update the shared latest_frames for the dashboard
                with frames_lock:
                    latest_frames[camera_id] = frame.copy()
            # No cv2.imshow or cv2.waitKey needed
            time.sleep(0.01)
    finally:
        notification_thread.stop()
        for thread in camera_threads:
            thread.stop()
        db_logger.close()
        # No cv2.destroyAllWindows()

if __name__ == "__main__":
    main()