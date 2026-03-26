"""
Gaze Tracker — Real-time gaze tracking with webcam.
Uses GazeFollower for precise screen-coordinate gaze estimation.

Configuration is loaded from a .env file (see .env.example).

Usage:
    python main.py                          # Run with visualizer
    python main.py --no-visualizer          # Status window only
    python main.py --repeat --cooldown 5    # Replay alert every 5s while looking away
    python main.py --recalibrate            # Force recalibration
"""

import argparse
import threading
import time
import sys
import os

import cv2
import numpy as np

try:
    from gazefollower import GazeFollower
except ImportError:
    print("Error: gazefollower not installed. Run: pip install gazefollower")
    sys.exit(1)


# ---------------------------------------------------------------------------
# GazeState — thread-safe container for latest gaze data
# ---------------------------------------------------------------------------

class HeadPoseEstimator:
    """Estimates head yaw/pitch from MediaPipe face landmarks using solvePnP."""

    # 3D model points of key facial landmarks (generic face model)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye left corner
        (43.3, 32.7, -26.0),      # Right eye right corner
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1),     # Right mouth corner
    ], dtype=np.float64)

    # MediaPipe face mesh landmark indices for the above points
    LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

    def __init__(self, img_w=640, img_h=480):
        self.update_camera_matrix(img_w, img_h)

    def update_camera_matrix(self, img_w, img_h):
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, landmarks, img_w, img_h):
        """
        Estimate head pose from 478-point MediaPipe face landmarks.
        Returns (yaw, pitch, roll) in degrees, or None if estimation fails.
        """
        if landmarks is None or len(landmarks) < 292:
            return None

        # Extract 2D image points for the key landmarks
        image_points = np.array([
            (landmarks[idx][0] * img_w, landmarks[idx][1] * img_h)
            if landmarks[idx][0] <= 1.0  # normalized coords
            else (landmarks[idx][0], landmarks[idx][1])  # pixel coords
            for idx in self.LANDMARK_IDS
        ], dtype=np.float64)

        # Update camera matrix if image size changed
        self.update_camera_matrix(img_w, img_h)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.MODEL_POINTS, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Decompose rotation matrix to Euler angles (ZYX convention)
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        yaw = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        pitch = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
        roll = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))

        return yaw, pitch, roll


class GazeState:
    # Head pose thresholds (degrees) — beyond these = looking away
    YAW_THRESHOLD = 30.0
    PITCH_THRESHOLD = 25.0

    def __init__(self, screen_w, screen_h, margin=50):
        self._lock = threading.Lock()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.margin = margin

        # Gaze data
        self.x = 0.0
        self.y = 0.0
        self.on_screen = False
        self.tracking = False
        self.tracking_state = None
        self.left_openness = 0.0
        self.right_openness = 0.0

        # Head pose data
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # Face data
        self.face_rect = None
        self.left_eye_rect = None
        self.right_eye_rect = None

        # Frame data (from camera callback)
        self.frame = None
        self.frame_timestamp = 0

        # State transition tracking (for alert sound)
        self._was_on_screen = False
        self._looking_away_since = None  # timestamp when looking away started
        self._looking_back_since = None  # timestamp when looking back started
        self._alert_fired = False  # whether alert already fired for this away period
        self._last_alert_time = 0  # timestamp of last alert played

        # Head pose estimator
        self._head_pose = HeadPoseEstimator()
        self._debug_frame_count = 0

    def on_gaze_update(self, face_info, gaze_info):
        """Subscriber callback — called from GazeFollower's camera thread."""
        with self._lock:
            # Face detection data
            if face_info is not None:
                self.face_rect = getattr(face_info, 'face_rect', None)
                self.left_eye_rect = getattr(face_info, 'left_rect', None)
                self.right_eye_rect = getattr(face_info, 'right_rect', None)

            self.tracking_state = getattr(gaze_info, 'tracking_state', None) if gaze_info else None

            # Extract gaze coordinates if available
            coords = None
            if gaze_info is not None and gaze_info.status:
                coords = gaze_info.filtered_gaze_coordinates
                self.left_openness = getattr(gaze_info, 'left_openness', 0.0)
                self.right_openness = getattr(gaze_info, 'right_openness', 0.0)

            if coords is not None:
                self.x = float(coords[0])
                self.y = float(coords[1])

            # Head pose based "looking away" detection
            # Run this whenever face_info has landmarks, even without gaze coords
            face_detected = False
            landmarks = getattr(face_info, 'face_landmarks', None) if face_info else None
            if landmarks is not None and len(landmarks) >= 292:
                face_detected = True
                img_w = getattr(face_info, 'img_w', 640)
                img_h = getattr(face_info, 'img_h', 480)

                head_on_screen = True
                pose = self._head_pose.estimate(landmarks, img_w, img_h)
                if pose is not None:
                    self.yaw, self.pitch, self.roll = pose
                    head_on_screen = (
                        abs(self.yaw) < self.YAW_THRESHOLD
                        and abs(self.pitch) < self.PITCH_THRESHOLD
                    )
                    # Debug: print angles every 30 frames
                    self._debug_frame_count += 1
                    if self._debug_frame_count % 30 == 0:
                        status = "ON" if head_on_screen else "AWAY"
                        print(f"[HEAD] yaw={self.yaw:.1f} pitch={self.pitch:.1f} roll={self.roll:.1f} → {status}")

                self.tracking = True
                self.on_screen = head_on_screen
            else:
                self.tracking = False
                self.on_screen = False

            # Track how long user has been looking away / back
            if not self.on_screen:
                if self._looking_away_since is None:
                    self._looking_away_since = time.time()
                self._looking_back_since = None
            else:
                self._looking_away_since = None
                if self._looking_back_since is None:
                    self._looking_back_since = time.time()

    def on_camera_frame(self, camera_running_state, timestamp, frame):
        """Camera image callback — stores the latest frame for the visualizer."""
        with self._lock:
            if isinstance(frame, np.ndarray):
                self.frame = frame.copy()
            self.frame_timestamp = timestamp

    def get_state(self):
        """Thread-safe snapshot of current state."""
        with self._lock:
            return {
                'x': self.x,
                'y': self.y,
                'on_screen': self.on_screen,
                'tracking': self.tracking,
                'tracking_state': self.tracking_state,
                'left_openness': self.left_openness,
                'right_openness': self.right_openness,
                'yaw': self.yaw,
                'pitch': self.pitch,
                'roll': self.roll,
                'face_rect': list(self.face_rect) if self.face_rect is not None else None,
                'left_eye_rect': list(self.left_eye_rect) if self.left_eye_rect is not None else None,
                'right_eye_rect': list(self.right_eye_rect) if self.right_eye_rect is not None else None,
            }

    def get_frame(self):
        """Thread-safe copy of the latest camera frame."""
        with self._lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def should_alert(self, away_duration, repeat=False, cooldown=10.0):
        """
        Returns True when user has been looking away for away_duration seconds.
        By default fires once per away-period. If repeat=True, fires again
        every `cooldown` seconds after the initial alert.
        """
        with self._lock:
            if self._looking_away_since is None:
                return False
            elapsed = time.time() - self._looking_away_since
            if elapsed < away_duration:
                return False
            # First alert for this away period
            if not self._alert_fired:
                self._alert_fired = True
                self._last_alert_time = time.time()
                return True
            # Repeat alert if enabled
            if repeat and time.time() - self._last_alert_time >= cooldown:
                self._last_alert_time = time.time()
                return True
            return False

    def should_stop_alert(self, look_back_duration=1.5):
        """Returns True when user has been looking at screen for look_back_duration seconds
        after an alert was fired."""
        with self._lock:
            if self._looking_back_since is None:
                return False
            if not self._alert_fired:
                return False
            if time.time() - self._looking_back_since >= look_back_duration:
                self._alert_fired = False
                return True
            return False


# ---------------------------------------------------------------------------
# AlertSound — plays MP3 when user looks away
# ---------------------------------------------------------------------------

class AlertSound:
    def __init__(self, sound_path):
        self.sound_path = sound_path
        self._initialized = False

        if not os.path.isfile(sound_path):
            print(f"Warning: Alert sound file not found: {sound_path}")
            return

        try:
            import pygame
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(sound_path)
            self._initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize audio: {e}")

    def play(self):
        """Play the alert sound."""
        if self._initialized:
            self._sound.play()

    def stop(self):
        """Stop the alert sound."""
        if self._initialized:
            self._sound.stop()

    def cleanup(self):
        if self._initialized:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GazeVisualizer — OpenCV window with camera feed + gaze overlay
# ---------------------------------------------------------------------------

class GazeVisualizer:
    WINDOW_NAME = "Gaze Tracker - Visualizer"

    def __init__(self, gaze_state):
        self.gaze_state = gaze_state
        self._fps_times = []
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, 640, 480)
            self._window_created = True
            print("OpenCV visualizer window created.")

    def _calc_fps(self):
        now = time.time()
        self._fps_times.append(now)
        # Keep last 30 timestamps
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        return len(self._fps_times)

    def draw_frame(self):
        """Draw one frame of the visualizer. Returns False if window closed."""
        self._ensure_window()

        frame = self.gaze_state.get_frame()
        if frame is None:
            # Show a black placeholder while waiting for frames
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera frames...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            cv2.imshow(self.WINDOW_NAME, placeholder)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return False
            return True

        state = self.gaze_state.get_state()

        # Convert RGB to BGR for OpenCV display
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display = frame.copy()

        h, w = display.shape[:2]
        fps = self._calc_fps()

        # Draw face bounding box
        if state['face_rect'] is not None:
            try:
                rect = [int(v) for v in state['face_rect']]
                x, y, rw, rh = rect[0], rect[1], rect[2], rect[3]
                color = (0, 255, 0) if state['on_screen'] else (0, 0, 255)
                cv2.rectangle(display, (x, y), (x + rw, y + rh), color, 2)
            except (IndexError, ValueError, TypeError):
                pass

        # Draw eye bounding boxes
        for eye_rect in [state['left_eye_rect'], state['right_eye_rect']]:
            if eye_rect is not None:
                try:
                    rect = [int(v) for v in eye_rect]
                    x, y, rw, rh = rect[0], rect[1], rect[2], rect[3]
                    cv2.rectangle(display, (x, y), (x + rw, y + rh), (255, 200, 0), 1)
                except (IndexError, ValueError, TypeError):
                    pass

        # Status bar background
        cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)

        # Status text
        if not state['tracking']:
            status_text = "NO FACE DETECTED"
            status_color = (128, 128, 128)
        elif state['on_screen']:
            status_text = "ON SCREEN"
            status_color = (0, 255, 0)
        else:
            status_text = "LOOKING AWAY"
            status_color = (0, 0, 255)

        cv2.putText(display, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Gaze coordinates
        if state['tracking']:
            coord_text = f"Gaze: ({state['x']:.0f}, {state['y']:.0f})"
            cv2.putText(display, coord_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Head pose angles
        if state['tracking']:
            pose_text = f"Yaw:{state['yaw']:.0f} Pitch:{state['pitch']:.0f}"
            cv2.putText(display, pose_text, (w - 220, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # FPS
        cv2.putText(display, f"FPS: {fps}", (w - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mini screen representation (bottom-right corner)
        mini_w, mini_h = 160, 90
        mini_x = w - mini_w - 10
        mini_y = h - mini_h - 10
        cv2.rectangle(display, (mini_x, mini_y),
                      (mini_x + mini_w, mini_y + mini_h), (100, 100, 100), 2)
        cv2.rectangle(display, (mini_x + 1, mini_y + 1),
                      (mini_x + mini_w - 1, mini_y + mini_h - 1), (30, 30, 30), -1)

        if state['tracking']:
            # Map gaze coords to mini screen
            sw, sh = self.gaze_state.screen_w, self.gaze_state.screen_h
            dot_x = int(mini_x + (state['x'] / sw) * mini_w)
            dot_y = int(mini_y + (state['y'] / sh) * mini_h)
            dot_x = max(mini_x, min(dot_x, mini_x + mini_w))
            dot_y = max(mini_y, min(dot_y, mini_y + mini_h))
            dot_color = (0, 255, 0) if state['on_screen'] else (0, 0, 255)
            cv2.circle(display, (dot_x, dot_y), 5, dot_color, -1)

        cv2.imshow(self.WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Escape
            return False
        return True

    def cleanup(self):
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# GazeApp — main application orchestrator
# ---------------------------------------------------------------------------

class GazeApp:
    def __init__(self, show_visualizer=True, alert_path=None,
                 away_duration=6.0, alert_repeat=False, alert_cooldown=10.0,
                 recalibrate=False):
        self.show_visualizer = show_visualizer
        self.alert_path = alert_path
        self.away_duration = away_duration
        self.alert_repeat = alert_repeat
        self.alert_cooldown = alert_cooldown
        self.recalibrate = recalibrate

        self.gaze_follower = None
        self.gaze_state = None
        self.visualizer = None
        self.alert = None

    def run(self):
        print("Initializing GazeFollower...")
        try:
            self.gaze_follower = GazeFollower()
        except Exception as e:
            print(f"Error initializing GazeFollower: {e}")
            print("Make sure your webcam is connected and camera access is allowed.")
            sys.exit(1)

        # Get screen size from GazeFollower's screen_size tuple (w, h)
        screen_size = getattr(self.gaze_follower, 'screen_size', None)
        if screen_size is not None:
            screen_w, screen_h = screen_size
        else:
            screen_w, screen_h = 1920, 1080

        print(f"Screen size: {screen_w}x{screen_h}")

        self.gaze_state = GazeState(screen_w, screen_h)

        # Set up visualizer
        if self.show_visualizer:
            self.visualizer = GazeVisualizer(self.gaze_state)
            # Patch the camera's stored callback to intercept frames.
            # We can't just replace gaze_follower.process_frame because
            # the camera holds a reference to the original bound method.
            camera = self.gaze_follower.camera
            original_callback = camera.callback_func

            def wrapped_callback(state, timestamp, frame, *args, **kwargs):
                # Capture the frame for our visualizer
                if isinstance(frame, np.ndarray):
                    with self.gaze_state._lock:
                        self.gaze_state.frame = frame.copy()
                        self.gaze_state.frame_timestamp = timestamp
                # Call the original processing (suppress calibration errors)
                try:
                    return original_callback(state, timestamp, frame, *args, **kwargs)
                except Exception:
                    pass

            camera.callback_func = wrapped_callback
            print("Camera frame interception set up for visualizer.")

        # Monkey-patch calibration.predict to handle broken saved models
        # GazeFollower has a bug where saved SVR models can become incompatible
        calibration = self.gaze_follower.calibration
        original_predict = calibration.predict
        _predict_warned = False

        def safe_predict(features, estimated_coordinate):
            nonlocal _predict_warned
            try:
                return original_predict(features, estimated_coordinate)
            except cv2.error:
                if not _predict_warned:
                    print("Warning: Saved calibration model is incompatible. "
                          "Run with --recalibrate to fix. Using uncalibrated gaze.")
                    _predict_warned = True
                return True, estimated_coordinate

        calibration.predict = safe_predict

        # Check if calibration already exists
        has_saved_calibration = getattr(calibration, 'has_calibrated', False)

        if has_saved_calibration and not self.recalibrate:
            print("\nUsing saved calibration. Run with --recalibrate to redo.")
        else:
            # Phase 1: Camera preview
            print("\n--- Camera Preview ---")
            print("Position yourself in front of the webcam.")
            print("Press any key to continue to calibration...")
            try:
                self.gaze_follower.preview()
            except Exception as e:
                print(f"Preview error: {e}")

            # Phase 2: Calibration
            print("\n--- Calibration ---")
            print("Look at each calibration point as it appears.")
            try:
                self.gaze_follower.calibrate()
                # Save calibration for future runs
                if hasattr(calibration, 'save_model'):
                    calibration.save_model()
                    print("Calibration saved for future sessions.")
            except Exception as e:
                print(f"Calibration error: {e}")
                print("Continuing without calibration (accuracy will be reduced).")

        # Close pygame windows left over from preview/calibration
        try:
            import pygame
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

        # Set up alert sound (AFTER pygame cleanup so mixer isn't killed)
        if self.alert_path:
            self.alert = AlertSound(self.alert_path)

        # Phase 3: Start tracking
        print("\n--- Tracking Started ---")
        print("Press Escape in the visualizer window to quit.")
        self.gaze_follower.add_subscriber(self.gaze_state.on_gaze_update)
        self.gaze_follower.start_sampling()
        print("Sampling started, launching visualizer...")

        # Phase 4: Main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        # Phase 5: Cleanup
        self._cleanup()

    def _main_loop(self):
        """Main update loop."""
        while True:
            # Check for look-away alert (after N consecutive seconds)
            if self.alert and self.gaze_state.should_alert(
                    self.away_duration, self.alert_repeat, self.alert_cooldown):
                self.alert.play()

            # Stop alert when user looks back for 1.5s
            if self.alert and self.gaze_state.should_stop_alert(1.5):
                self.alert.stop()

            # Update visualizer
            if self.visualizer:
                if not self.visualizer.draw_frame():
                    break  # Window closed
            else:
                # No visualizer — just print status periodically
                state = self.gaze_state.get_state()
                if state['tracking']:
                    status = "ON SCREEN" if state['on_screen'] else "LOOKING AWAY"
                    print(f"\r{status} - Gaze: ({state['x']:.0f}, {state['y']:.0f})", end="")
                else:
                    print(f"\rNO FACE DETECTED", end="")
                time.sleep(0.1)

    def _cleanup(self):
        print("\n\nShutting down...")
        try:
            self.gaze_follower.stop_sampling()
        except Exception:
            pass

        try:
            self.gaze_follower.save_data("gaze_session.csv")
            print("Session data saved to gaze_session.csv")
        except Exception as e:
            print(f"Could not save session data: {e}")

        try:
            self.gaze_follower.release()
        except Exception:
            pass

        if self.visualizer:
            self.visualizer.cleanup()

        if self.alert:
            self.alert.cleanup()

        print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Gaze Tracker — real-time gaze tracking")
    parser.add_argument('--no-visualizer', action='store_true',
                        help='Disable the camera feed visualizer')
    parser.add_argument('--repeat', action='store_true',
                        help='Replay alert sound while still looking away (with cooldown)')
    parser.add_argument('--cooldown', type=float, default=10.0,
                        help='Seconds between repeated alerts if --repeat is set (default: 10)')
    parser.add_argument('--away-duration', type=float, default=None,
                        help='Seconds of looking away before alert plays (default: 6)')
    parser.add_argument('--recalibrate', action='store_true',
                        help='Force recalibration even if saved calibration exists')
    args = parser.parse_args()

    # Load config from .env, CLI args override
    alert_path = os.environ.get('ALERT_PATH', '').strip() or None
    away_duration = args.away_duration or float(os.environ.get('AWAY_DURATION', '6'))

    if alert_path:
        print(f"Alert sound: {alert_path}")
    else:
        print("No alert sound configured. Set ALERT_PATH in .env to enable.")

    app = GazeApp(
        show_visualizer=not args.no_visualizer,
        alert_path=alert_path,
        away_duration=away_duration,
        alert_repeat=args.repeat,
        alert_cooldown=args.cooldown,
        recalibrate=args.recalibrate,
    )
    app.run()


if __name__ == '__main__':
    main()
