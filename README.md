# Pedestrian Camera Detection - YOLO AI

A real-time multi-camera surveillance system that uses YOLO object detection and tracking to monitor people entering and leaving camera frames. The system provides live video feeds, event logging, and Telegram notifications.

## Features

- **Multi-Camera Support**: Monitor multiple cameras simultaneously
- **YOLO Detection**: Uses YOLO (You Only Look Once) for real-time person detection
- **Event Logging**: Records entry/exit events with timestamps and duration
- **Real-time Dashboard**: Web-based dashboard with live video feeds and event logs
- **Telegram Notifications**: Instant alerts when people enter or leave camera frames
- **Database Storage**: SQLite database for persistent event logging
- **Exit Buffer System**: Prevents false exit events with configurable buffer time

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- Webcam(s) or video files
- Telegram Bot Token (for notifications)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pedestrian_detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model**:
   The system uses YOLO11n by default. The model will be automatically downloaded on first run, or you can manually download it.

4. **Configure Telegram notifications** (optional):
   - Copy `config_example.py` to `config.py`
   - Add your Telegram bot token and chat ID

## Configuration

### Basic Configuration

Create a `config.py` file based on `config_example.py`:

```python
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

### Camera Configuration

In `app.py`, modify the `INPUT_MODE` variable:

```python
INPUT_MODE = "CAMERA"  # For real cameras
# or
INPUT_MODE = "VIDEO"   # For video files
```

For video mode, update the `input_sources` list with your video file paths:

```python
input_sources = [
    "path/to/video1.mp4",
    "path/to/video2.mp4",
    "path/to/video3.mp4",
]
```

## Usage

### Starting the System

Run the main application:

```bash
python app.py
```

The system will:
1. Initialize camera threads for each input source
2. Start the web dashboard on `http://localhost:5000`
3. Begin monitoring and tracking people
4. Send Telegram notifications for events

### Accessing the Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

The dashboard provides:
- Live video feeds from all cameras
- Real-time event logs
- Person count per camera
- Event timestamps and durations

### Core Components

1. **CameraThread**: Handles individual camera/video processing
2. **PersonTracker**: Manages person tracking and event detection
3. **DatabaseLogger**: Handles event logging to SQLite database
4. **TelegramNotifier**: Sends notifications via Telegram
5. **Flask Dashboard**: Web interface for monitoring


## Customization

### Adjusting Exit Buffer Time

Modify the `exit_buffer_seconds` parameter in the `PersonTracker` class:

```python
self.person_tracker = PersonTracker(camera_id, exit_buffer_seconds=5)
```

### Changing YOLO Model

Update the `model_path` variable in `main()`:

```python
model_path = "yolov8n.pt"  # or any other YOLO model
```

### Database Configuration

The system automatically creates a `logs.db` SQLite database. You can modify the database path in the `DatabaseLogger` class. 