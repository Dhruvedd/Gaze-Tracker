# Gaze Tracker

A desktop app that uses your webcam to detect when you're looking away from your screen — and nudges you back with an alert sound.

## The Problem

Whether you're studying, working remotely, or trying to stay locked in during a long coding session, attention drift is real. You glance at your phone, look out the window, or zone out — and minutes slip by before you notice.

Gaze Tracker sits in the background watching your head orientation through your webcam. When you look away from the screen for too long, it plays an alert sound to bring your focus back. Think of it as an automated accountability partner for your attention span.

## How It Works

1. Your webcam captures a live video feed
2. **MediaPipe Face Mesh** detects 478 facial landmarks in each frame
3. **Head pose estimation** (via `cv2.solvePnP`) calculates your yaw and pitch angles from 6 key landmarks
4. If your head is turned beyond the threshold for a configurable number of seconds, the app plays an MP3 alert
5. When you look back at the screen for 1.5 seconds, the alert stops

The app also uses [GazeFollower](https://github.com/GanchengZhu/GazeFollower) for gaze calibration and screen-coordinate tracking, displayed in the visualizer.

## Features

- **Real-time visualizer** — OpenCV window showing your camera feed with face/eye bounding boxes, status bar, gaze coordinates, and a mini screen map
- **Head pose detection** — Yaw/pitch-based "looking away" classification that works regardless of gaze calibration quality
- **Configurable alert sound** — Point it at any MP3 file; plays after N seconds of looking away, stops when you look back
- **Calibration persistence** — Calibrate once, skip it on future runs (saved to `~/GazeFollower/calibration/`)
- **Customizable thresholds** — Adjust away duration, enable/disable alert repeat, set cooldown timers

## Installation

**Prerequisites:** Python 3.11+, a webcam

```bash
git clone https://github.com/yourusername/Gaze-Tracker.git
cd Gaze-Tracker

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Open `.env` and set:

```env
# Path to the MP3 file to play when you look away (leave empty to disable alerts)
ALERT_PATH=C:\path\to\your\alert.mp3

# Seconds of looking away before the alert plays (default: 6)
AWAY_DURATION=6
```

## Usage

**First run** (requires calibration):

```bash
python main.py
```

A pygame window will open for camera preview and calibration. Look at each dot as it appears on screen. After calibration, the OpenCV visualizer window will launch.

**Subsequent runs** (uses saved calibration):

```bash
python main.py
```

**Force recalibration:**

```bash
python main.py --recalibrate
```

**All CLI options:**

| Flag | Description |
|------|-------------|
| `--recalibrate` | Force recalibration even if saved calibration exists |
| `--no-visualizer` | Disable the camera feed window (text-only status) |
| `--away-duration N` | Override seconds before alert triggers (default: from `.env` or 6) |
| `--repeat` | Replay alert sound while still looking away |
| `--cooldown N` | Seconds between repeated alerts when `--repeat` is set (default: 10) |

**Quit:** Press `Escape` in the visualizer window, or `Ctrl+C` in the terminal.

## Visualizer

The OpenCV window shows:

- **Status bar** (top) — "ON SCREEN" (green), "LOOKING AWAY" (red), or "NO FACE DETECTED" (gray)
- **Face bounding box** — Green when on screen, red when looking away
- **Eye bounding boxes** — Shown in yellow
- **Gaze coordinates and head pose angles** — Displayed in the status bar
- **Mini screen map** (bottom-right) — A small rectangle representing your screen with a dot showing estimated gaze position
