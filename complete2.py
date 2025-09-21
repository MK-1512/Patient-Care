# -*- coding: utf-8 -*-
"""
RE-MODIFIED: complete2.py (Final Gesture Names - Indentation Corrected AGAIN)

Changes:
- Mapped 'ILoveYou' gesture to "Call Family".
- Mapped 'None' gesture category to "Unrecognized Gesture" for display.
- Prevented "Unrecognized Gesture" from being spoken aloud.
- Kept hybrid sad detection logic and FaceLandmarker integration.
- Corrected previous indentation errors.
"""
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import threading
import time
import sqlite3
import sys
from datetime import datetime
import pytz
import os

# --- Add OpenCV Build Info Print ---
try:
    print("--- OpenCV Build Information ---")
    # print(cv2.getBuildInformation()) # Can be very verbose, uncomment if needed
    print(f"OpenCV Version: {cv2.__version__}")
    print("------------------------------")
except Exception as e:
    print(f"Could not get OpenCV build info: {e}")
# ------------------------------------

# MediaPipe Tasks imports
from mediapipe.framework.formats import landmark_pb2
try:
    # Import necessary components from mediapipe.tasks
    from mediapipe.tasks import python as mp_python_task
    from mediapipe.tasks.python import vision as mp_vision_task
    from mediapipe import ImageFormat # For mp.Image format specifier
    TASKS_AVAILABLE = True
except ImportError:
    print("Warning: mediapipe.tasks not found. Gesture and new Face recognition disabled.")
    TASKS_AVAILABLE = False; mp_python_task = None; mp_vision_task = None; ImageFormat = None
# Plyer import
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    print("Warning: 'plyer' library not found. Desktop notifications disabled.")
    PLYER_AVAILABLE = False; notification = None
# PyQt5 import
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem
    PYQT5_AVAILABLE = True
except ImportError:
    print("Warning: PyQt5 not found. GUI log viewer disabled.")
    PYQT5_AVAILABLE = False

# --- Database Setup ---
conn = None; cursor = None; BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "activity_log.db"); MEDIA_DIR = os.path.join(BASE_DIR, "media")
RECORDINGS_DIR = os.path.join(MEDIA_DIR, "recordings"); db_connection_error = False
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (timestamp TEXT, type TEXT, content TEXT )''')
    conn.commit()
    print("Database connection successful.")
except sqlite3.Error as db_setup_err:
    print(f"CRITICAL ERROR setting up database: {db_setup_err}")
    db_connection_error = True; conn = None; cursor = None

# --- Helper Functions ---
def log_event(event_type, content):
    global conn, cursor
    if not cursor or not conn: return
    try:
        try:
            local_tz = pytz.timezone('Asia/Kolkata')
            timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z%z")
        except Exception:
            timestamp = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z")
        log_cursor = conn.cursor()
        log_cursor.execute("INSERT INTO logs VALUES (?, ?, ?)", (timestamp, event_type, content))
        conn.commit()
    except Exception as e:
        print(f"DB Log Error: {e}")

def speak(text):
    # *** MODIFIED: Prevent speaking "Unrecognized Gesture" ***
    if not text or text in ['None', 'Unrecognized Gesture']: return # Don't speak None or the generic term
    def run():
        try:
            if sys.platform == 'darwin':
                subprocess.call(['say', text])
            elif sys.platform.startswith('linux'):
                subprocess.call(['espeak', '-v', 'en+f3', '-s', '160', text])
            else:
                print(f"Speech not configured for OS: {sys.platform}")
        except FileNotFoundError:
            print("Error: Speech command not found.")
        except Exception as e:
            print(f"Speech Error: {e}")
    threading.Thread(target=run, daemon=True).start()

# --- Draw landmarks (accepts FaceLandmarker results) ---
def draw_landmarks_on_image(bgr_image, hands_result=None, pose_result=None, face_result=None):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh_module = mp.solutions.face_mesh # Still needed for connections
    mp_face_connections = mp_face_mesh_module.FACEMESH_TESSELATION

    # Hand Drawing
    if hands_result and hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Pose Drawing
    if pose_result and pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            bgr_image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Face Drawing (using FaceLandmarker landmarks)
    if face_result and face_result.face_landmarks:
        for face_landmarks_list in face_result.face_landmarks:
            proto_landmarks = landmark_pb2.NormalizedLandmarkList()
            proto_landmarks.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks_list
            ])
            mp_drawing.draw_landmarks(
                image=bgr_image,
                landmark_list=proto_landmarks,
                connections=mp_face_connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    return bgr_image


# --- Geometric Sadness Check (based on old logic) ---
def check_geometric_sad(landmarks, w, h):
    """
    Checks for sadness based on vertical distance between mouth corners and lip centers.
    Returns True if sad geometry detected, False otherwise.
    Uses landmark indices relevant to FaceLandmarker (similar to old FaceMesh).
    """
    try:
        # Landmark indices (approximate mapping from FaceMesh to FaceLandmarker):
        # 61: Left mouth corner
        # 291: Right mouth corner
        # 13: Upper lip center-top
        # 14: Lower lip center-bottom
        if landmarks and len(landmarks) > 291: # Ensure enough landmarks exist
            l = landmarks[61]
            r = landmarks[291]
            t = landmarks[13]
            b = landmarks[14]

            if all([l, r, t, b]): # Check if landmarks were detected
                # Calculate vertical distance, scaled by frame height
                # Positive diff means corners are lower than lip centers
                diff = ((l.y + r.y) / 2 - (t.y + b.y) / 2) * h
                # print(f"Geometric Diff: {diff:.2f}") # Optional debug print
                if diff > 4.5: # Original threshold for sadness
                    return True
        return False # Not sad, or error, or not enough landmarks
    except IndexError:
        # print(f"IndexError in check_geometric_sad. Landmarks length: {len(landmarks)}")
        return False
    except Exception as e:
        print(f"Geometric Sad Check Err: {e}")
        return False


# --- Blendshape Expression Check ---
def get_expression_from_blendshapes(blendshapes):
    """
    Analyzes blendshapes from FaceLandmarker result.
    Returns 'happy', 'sad_blendshape', or 'neutral'.
    DEBUG PRINT ENABLED.
    """
    if not blendshapes or len(blendshapes) == 0:
        return "neutral" # Treat as neutral if no blendshapes

    categories = blendshapes[0]
    blendshape_dict = {category.category_name: category.score for category in categories}

    # Thresholds (lowered previously)
    smile_threshold = 0.4
    frown_threshold = 0.20
    brow_down_threshold = 0.15

    # Get relevant scores
    avg_smile = (blendshape_dict.get('mouthSmileLeft', 0) + blendshape_dict.get('mouthSmileRight', 0)) / 2
    avg_frown = (blendshape_dict.get('mouthFrownLeft', 0) + blendshape_dict.get('mouthFrownRight', 0)) / 2
    avg_brow_down = (blendshape_dict.get('browDownLeft', 0) + blendshape_dict.get('browDownRight', 0)) / 2

    # --- DEBUG PRINT (ENABLED) ---
    print(f"Smile: {avg_smile:.2f}, Frown: {avg_frown:.2f}, BrowDown: {avg_brow_down:.2f}")
    # --- END DEBUG PRINT ---

    # --- Determine expression based ONLY on blendshapes ---
    if avg_smile > smile_threshold and avg_smile > avg_frown and avg_smile > avg_brow_down:
        return "happy"
    # Check for sadness based on blendshapes
    elif (avg_frown > frown_threshold or avg_brow_down > brow_down_threshold) and avg_smile < (smile_threshold * 0.8):
        return "sad_blendshape" # Indicate sadness detected via blendshapes
    else:
        return "neutral"


# --- Main Application Logic ---
def run_main_app():
    global conn, cursor
    if db_connection_error:
        print("Exiting: DB setup failed.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        log_event("error", "Video device fail")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0
    print(f"Cam: {frame_width}x{frame_height}, Rec FPS: {fps}")

    # Video Writer Setup
    video_writer = None; video_filename = ""
    try:
        print(f"Attempting to create recordings directory: {RECORDINGS_DIR}")
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        print(f"Directory exists or created: {RECORDINGS_DIR}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(RECORDINGS_DIR, f"rec_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f"Using FourCC: MJPG")
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        if video_writer.isOpened():
            print(f"VideoWriter opened successfully: {video_filename}")
            log_event("info", f"Rec Start: {video_filename}")
        else:
            print(f"ERROR: VideoWriter failed open: {video_filename}")
            log_event("error", f"VideoWriter fail open: {video_filename}")
            video_writer = None
    except OSError as ose:
        print(f"OSError VidSetup: {ose}")
        log_event("error", f"OSError VidSetup: {ose}")
        video_writer = None
    except Exception as e:
        print(f"VideoWriter Setup Err: {e}")
        log_event("error", f"VideoWriter setup fail: {e}")
        video_writer = None

    print("\nStarting ElderCare System...")

    # MediaPipe Setup
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # --- Initialize Tasks Recognizers/Landmarkers ---
    gesture_recognizer = None
    face_landmarker = None
    if TASKS_AVAILABLE:
        # Gesture Recognizer Setup
        try:
            gesture_model_path = os.path.join(BASE_DIR, 'gesture_recognizer.task')
            if not os.path.exists(gesture_model_path):
                raise FileNotFoundError(f"Gesture model not found: {gesture_model_path}")
            gesture_base_options = mp_python_task.BaseOptions(model_asset_path=gesture_model_path)
            gesture_options = mp_vision_task.GestureRecognizerOptions(base_options=gesture_base_options, running_mode=mp_vision_task.RunningMode.IMAGE, num_hands=2)
            gesture_recognizer = mp_vision_task.GestureRecognizer.create_from_options(gesture_options)
            print("Gesture recognizer created.")
        except Exception as e:
            print(f"Gesture Recognizer Err: {e}")
            log_event("error", f"Gesture Recognizer fail: {e}")

        # Face Landmarker Setup
        try:
            face_model_path = os.path.join(BASE_DIR, 'face_landmarker.task')
            if not os.path.exists(face_model_path):
                raise FileNotFoundError(f"Face model not found: {face_model_path}")
            face_base_options = mp_python_task.BaseOptions(model_asset_path=face_model_path)
            face_options = mp_vision_task.FaceLandmarkerOptions(base_options=face_base_options, running_mode=mp_vision_task.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
            face_landmarker = mp_vision_task.FaceLandmarker.create_from_options(face_options)
            print("Face landmarker created.")
        except Exception as e:
            print(f"Face Landmarker Err: {e}")
            log_event("error", f"Face Landmarker fail: {e}")
    else:
        print("WARN: MediaPipe Tasks not available.")


    # --- Context Managers for MediaPipe Solutions ---
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # *** FINAL GESTURE LISTS ***
        # Model output names (must match the model's categories)
        # Ensure 'None' is the last item if assigned_gestures has it last.
        default_gestures = ['Closed_Fist', 'Thumb_Down', 'Open_Palm', 'Victory', 'Pointing_Up', 'Thumb_Up', 'ILoveYou', 'None']
        # Spoken names (map 'ILoveYou' -> "Call Family", map 'None' -> "Unrecognized Gesture")
        assigned_gestures = ['Emergency', 'Not Good', 'Request Doctor', 'All set', 'Request Food', 'Well & Good', 'Call Family', 'Unrecognized Gesture']


        # --- State Variables ---
        mode = 'idle'; last_mode_message = ""
        hand_buffer = []; last_hand_spoken = 'None'; last_hand_time = 0
        expression_buffer = []; last_expression_spoken = 'None'; last_expression_time = 0
        last_motion_spoken = 'None'; last_motion_time = 0
        motion_threshold = 15
        prev_positions = {"head": None, "left_hip": None, "left_wrist": None, "right_wrist": None}
        frame_count = 0
        no_hand_start_time = None; no_face_start_time = None; no_motion_start_time = None
        played_no_hand_audio = False; played_no_face_audio = False; played_no_motion_audio = False; played_no_person_audio = False
        AUDIO_ALERT_DELAY = 3.0

        # --- Motion Helper ---
        def get_landmark_pos(landmarks_list, index):
            if landmarks_list and index < len(landmarks_list):
                lm = landmarks_list[index]
                if lm and hasattr(lm, 'x') and hasattr(lm, 'y') and np.isfinite(lm.x) and np.isfinite(lm.y):
                    return np.array([lm.x * frame_width, lm.y * frame_height])
            return None

        # --- Main Loop ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERR: Cam frame fail")
                time.sleep(0.1)
                break

            output_frame = frame.copy()
            current_status_text = ""
            perform_processing = (mode != 'idle')

            hands_result, pose_result, face_result = None, None, None # Reset results

            if perform_processing:
                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = None # For Tasks API
                if TASKS_AVAILABLE and ImageFormat:
                    try:
                        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                    except Exception as img_conv_err:
                        print(f"Error creating mp.Image: {img_conv_err}")
                        log_event("error", f"mp.Image creation failed: {img_conv_err}")
                        perform_processing = False

                rgb_frame.flags.writeable = False # For solutions API
                try:
                    # Run Solutions API models
                    hands_result = hands.process(rgb_frame)
                    pose_result = pose.process(rgb_frame)
                except Exception as process_err:
                    print(f"MP Process Err: {process_err}")
                    log_event("error", f"MP Process Err: {process_err}")
                rgb_frame.flags.writeable = True
            else:
                current_status_text = "Select Mode: 1:Hand 2:Face 3:Motion"
                if last_mode_message != current_status_text:
                    print("\nIdle: Select Mode [1,2,3] or q")
                    last_mode_message = current_status_text


            if perform_processing:
                last_mode_message = ""
                current_time = time.time()

                # --- Hand Gesture Mode (Uses updated gesture lists & logic) ---
                if mode == 'hand':
                    status_set = False
                    hands_on_frame = hands_result and hands_result.multi_hand_landmarks
                    # Default gesture if detection fails or model returns None/Unknown
                    gesture = 'Unrecognized Gesture' # Default to the new generic term

                    if not hands_on_frame:
                        # If no hand is detected, show "No hand detected"
                        current_status_text = "No hand detected"
                        # Use the generic term for internal buffer consistency
                        gesture = 'Unrecognized Gesture'
                        if last_hand_spoken != "No hand detected":
                            last_hand_spoken = "No hand detected"; hand_buffer.clear() # Reset buffer, display text
                        if no_hand_start_time is None:
                            no_hand_start_time = current_time
                        elif (current_time - no_hand_start_time) > AUDIO_ALERT_DELAY and not played_no_hand_audio:
                            speak("no hand detected"); log_event("audio_alert", "no hand detected (3s)"); played_no_hand_audio = True
                        hand_buffer.append(gesture) # Add 'Unrecognized Gesture' to buffer
                        if len(hand_buffer)>5: hand_buffer.pop(0)
                        status_set = True
                    elif not gesture_recognizer:
                        current_status_text = "Gesture Recognizer Failed/Disabled"
                        no_hand_start_time = None; played_no_hand_audio = False; status_set = True
                    elif mp_image is None:
                        current_status_text = "Error creating image for gesture detection"
                        no_hand_start_time = None; played_no_hand_audio = False; status_set = True
                    else:
                        no_hand_start_time = None; played_no_hand_audio = False
                        try:
                            gesture_recognition_result = gesture_recognizer.recognize(mp_image)
                            # Process result if gestures are found
                            if gesture_recognition_result.gestures and gesture_recognition_result.gestures[0]:
                                name = gesture_recognition_result.gestures[0][0].category_name
                                if name in default_gestures:
                                    try:
                                        idx = default_gestures.index(name)
                                        gesture = assigned_gestures[idx] # Map to assigned name ('Call Family', 'Unrecognized Gesture', etc.)
                                    except (ValueError, IndexError):
                                        gesture = "Unknown Category Error" # Should not happen
                                elif name and name != 'None': # Handle unknown but named category from model
                                    gesture = f"Unknown ({name})"
                                # If name is 'None' from model, 'gesture' remains 'Unrecognized Gesture'

                            hand_buffer.append(gesture) # Add the resulting *assigned* name to buffer
                            if len(hand_buffer)>5: hand_buffer.pop(0)
                            stable = max(set(hand_buffer), key=hand_buffer.count) if hand_buffer else 'Unrecognized Gesture'

                            # Only speak and notify for specific, meaningful gestures
                            meaningful_gestures = [g for g in assigned_gestures if g != 'Unrecognized Gesture'] # List of meaningful terms
                            if stable in meaningful_gestures and (stable != last_hand_spoken or (current_time - last_hand_time) > 3):
                                print(f"Gesture: {stable}"); speak(stable); log_event("gesture", stable)
                                if PLYER_AVAILABLE:
                                    try:
                                        notification.notify(title='Gesture', message=f"Detected: {stable}", timeout=10)
                                    except Exception as e:
                                        print(f"Notify Err:{e}")
                                last_hand_spoken = stable; last_hand_time = current_time
                            # Update last spoken regardless, for display consistency
                            elif stable != last_hand_spoken:
                                 last_hand_spoken = stable

                            # Update status text for display (show the stable result)
                            current_status_text = f"Gesture: {stable}"
                            status_set = True
                            output_frame = draw_landmarks_on_image(output_frame, hands_result=hands_result) # Draw based on MP Hands result
                        except Exception as gesture_err:
                            print(f"Gesture Err:{gesture_err}")
                            log_event("error", f"Gesture process fail: {gesture_err}")
                            current_status_text = "Gesture Error"; status_set = True
                    if not status_set:
                        current_status_text="Gesture: Init..."
                        no_hand_start_time = None; played_no_hand_audio = False


                # --- Face Expression Mode (HYBRID APPROACH) ---
                elif mode == 'face':
                    expr = "Detecting..."
                    current_frame_expression = "neutral" # Default for the current frame

                    if not face_landmarker:
                        expr = "Face Landmarker Failed/Disabled"
                        no_face_start_time = None; played_no_face_audio = False
                    elif mp_image is None:
                        expr = "Error creating image"
                        no_face_start_time = None; played_no_face_audio = False
                    else:
                        try:
                            face_result = face_landmarker.detect(mp_image)
                            face_present = face_result and face_result.face_landmarks and len(face_result.face_landmarks) > 0

                            if face_present:
                                no_face_start_time = None; played_no_face_audio = False
                                blendshape_expr = get_expression_from_blendshapes(face_result.face_blendshapes)
                                geometric_sad_detected = check_geometric_sad(face_result.face_landmarks[0], frame_width, frame_height)

                                if blendshape_expr == "happy":
                                    current_frame_expression = "happy"
                                elif blendshape_expr == "sad_blendshape" or geometric_sad_detected:
                                    current_frame_expression = "sad"
                                else:
                                    current_frame_expression = "neutral"

                                expression_buffer.append(current_frame_expression)
                                if len(expression_buffer)>7: expression_buffer.pop(0)

                                if len(expression_buffer)>=5:
                                    majority = max(set(expression_buffer), key=expression_buffer.count)
                                    if expression_buffer.count(majority)>=5:
                                        expr = majority.capitalize()
                                        if expr!=last_expression_spoken or (current_time-last_expression_time)>3:
                                            print(f"Expression: {expr}"); speak(expr); log_event("expression",expr)
                                            if PLYER_AVAILABLE:
                                                try:
                                                    notification.notify(title='Expression', message=f"Detected: {expr}", timeout=8)
                                                except Exception as e:
                                                    print(f"Notify Err:{e}")
                                            last_expression_spoken=expr; last_expression_time=current_time
                                    else:
                                        expr = "Analyzing..." if last_expression_spoken in ["Detecting...","Analyzing..."] else last_expression_spoken.capitalize()
                                else:
                                    expr = "Analyzing..."
                                output_frame = draw_landmarks_on_image(output_frame, face_result=face_result)
                            else: # No face detected
                                expr="No face"
                                current_frame_expression = "neutral"
                                expression_buffer.append(current_frame_expression)
                                if len(expression_buffer)>7: expression_buffer.pop(0)
                                if last_expression_spoken != "No face":
                                    expression_buffer.clear(); last_expression_spoken="No face"
                                if no_face_start_time is None:
                                    no_face_start_time = current_time
                                elif (current_time - no_face_start_time) > AUDIO_ALERT_DELAY and not played_no_face_audio:
                                    speak("no face detected"); log_event("audio_alert", "no face detected (3s)"); played_no_face_audio = True
                        except Exception as face_err:
                            print(f"Face Hybrid Err:{face_err}")
                            log_event("error", f"Face hybrid process fail: {face_err}")
                            expr="Error"; last_expression_spoken="error"; current_frame_expression = "neutral"
                            expression_buffer.append(current_frame_expression)
                            if len(expression_buffer)>7: expression_buffer.pop(0)
                            if no_face_start_time is None: no_face_start_time = current_time
                    current_status_text = f"Expression: {expr}"

                # --- Motion Detection Mode (Unchanged) ---
                elif mode == 'motion':
                    motion_part = None; motion_detected_this_frame = False
                    debug_motion = (frame_count % 60 == 0)
                    pose_landmarks = pose_result.pose_landmarks.landmark if pose_result and pose_result.pose_landmarks else None
                    person_detected = bool(pose_landmarks or (hands_result and hands_result.multi_hand_landmarks))

                    if not person_detected:
                        current_status_text = "No person detected"
                        if not played_no_person_audio:
                            speak("no person detected"); log_event("audio_alert", "no person detected (immediate)"); played_no_person_audio = True
                        if last_motion_spoken != "No person detected":
                            last_motion_spoken = "No person detected"; prev_positions = {k: None for k in prev_positions}
                        no_motion_start_time = None; played_no_motion_audio = False
                    else:
                        played_no_person_audio = False; current_motion_description = None
                        if debug_motion: print(f"--- Motion Debug {frame_count} ---")
                        # Head Motion
                        if pose_landmarks:
                            head_lm_idx=mp_pose.PoseLandmark.NOSE
                            head_pos=get_landmark_pos(pose_landmarks,head_lm_idx)
                            head_vis = pose_landmarks[head_lm_idx].visibility if head_lm_idx<len(pose_landmarks) and pose_landmarks[head_lm_idx] else 0
                            if debug_motion: print(f"  Head: Pos={'Y' if head_pos is not None else 'N'}, Vis={head_vis:.2f}", end='')
                            if head_pos is not None:
                                prev=prev_positions.get("head")
                                if prev is not None:
                                    dist=np.linalg.norm(head_pos-prev)
                                    if debug_motion: print(f", Prv=Y, D={dist:.1f}", end='')
                                    if dist>motion_threshold:
                                        if not motion_detected_this_frame: current_motion_description="head movement detected"
                                        motion_detected_this_frame=True
                                elif debug_motion: print(", Prv=N", end='')
                                prev_positions["head"]=head_pos
                                if debug_motion: print(", Stored=Y")
                            elif debug_motion: print("")
                        # Body Motion
                        if pose_landmarks:
                            hip_lm_idx=mp_pose.PoseLandmark.LEFT_HIP
                            hip_pos=get_landmark_pos(pose_landmarks,hip_lm_idx)
                            hip_vis = pose_landmarks[hip_lm_idx].visibility if hip_lm_idx<len(pose_landmarks) and pose_landmarks[hip_lm_idx] else 0
                            if debug_motion: print(f"  LHip: Pos={'Y' if hip_pos is not None else 'N'}, Vis={hip_vis:.2f}", end='')
                            if hip_pos is not None:
                                prev=prev_positions.get("left_hip")
                                if prev is not None:
                                    dist=np.linalg.norm(hip_pos-prev)
                                    if debug_motion: print(f", Prv=Y, D={dist:.1f}", end='')
                                    if dist>motion_threshold:
                                        if not motion_detected_this_frame: current_motion_description="body movement detected"
                                        motion_detected_this_frame=True
                                elif debug_motion: print(", Prv=N", end='')
                                prev_positions["left_hip"]=hip_pos
                                if debug_motion: print(", Stored=Y")
                            elif debug_motion: print("")
                        # Hand Motion
                        if hands_result and hands_result.multi_hand_landmarks:
                             for i, hand_landmarks_obj in enumerate(hands_result.multi_hand_landmarks):
                                hand_landmarks=hand_landmarks_obj.landmark
                                hand_label=f"Hand{i}"; wrist_idx=mp_hands.HandLandmark.WRIST
                                wrist_pos=get_landmark_pos(hand_landmarks,wrist_idx)
                                handedness_list = hands_result.multi_handedness; side = "unknown"
                                if handedness_list and i < len(handedness_list): side = handedness_list[i].classification[0].label.lower()
                                else: side = "left" if wrist_pos is not None and wrist_pos[0] < frame_width / 2 else "right"
                                part_key = f"{side}_wrist"
                                if debug_motion: print(f"  {hand_label} ({side}) Wrist: Pos={'Y' if wrist_pos is not None else 'N'}", end='')
                                if wrist_pos is not None:
                                    prev=prev_positions.get(part_key)
                                    if prev is not None:
                                        dist=np.linalg.norm(wrist_pos-prev)
                                        if debug_motion: print(f", Prv=Y, D={dist:.1f}", end='')
                                        if dist>motion_threshold:
                                            if not motion_detected_this_frame: current_motion_description="hand movement detected"
                                            motion_detected_this_frame=True
                                    elif debug_motion: print(", Prv=N", end='')
                                    prev_positions[part_key]=wrist_pos
                                    if debug_motion: print(", Stored=Y")
                                elif debug_motion: print("")
                        # Update Motion Status
                        if motion_detected_this_frame and current_motion_description:
                            motion_part=current_motion_description
                            if motion_part!=last_motion_spoken or (current_time-last_motion_time)>3:
                                print(f"Motion: {motion_part}"); speak(motion_part); log_event("motion",motion_part)
                                if PLYER_AVAILABLE:
                                    try: notification.notify(title='Motion', message=f"{motion_part}", timeout=5)
                                    except Exception as e: print(f"Notify Err:{e}")
                                last_motion_spoken=motion_part; last_motion_time=current_time
                            current_status_text=f"Motion: {last_motion_spoken}"
                            no_motion_start_time=None; played_no_motion_audio=False
                        elif not motion_detected_this_frame:
                            if (current_time-last_motion_time)>AUDIO_ALERT_DELAY:
                                current_status_text="Motion: No motion detected"
                                if last_motion_spoken!="No motion detected":
                                    print("Motion: No motion"); log_event("motion","No motion detected"); last_motion_spoken="No motion detected"
                                if no_motion_start_time is None:
                                    no_motion_start_time=current_time
                                elif (current_time-no_motion_start_time)>AUDIO_ALERT_DELAY and not played_no_motion_audio:
                                    speak("no motion detected"); log_event("audio_alert","no motion detected (3s continuous)"); played_no_motion_audio=True
                            else:
                                current_status_text=f"Motion: {last_motion_spoken}" if last_motion_spoken not in ['None','No motion detected','No person detected'] else "Person detected"
                                no_motion_start_time=None; played_no_motion_audio=False
                        else:
                            no_motion_start_time=None; played_no_motion_audio=False

                    if pose_result and pose_result.pose_landmarks:
                         output_frame = draw_landmarks_on_image(output_frame, pose_result=pose_result)


            # --- Display Frame and Text Overlays ---
            flipped_display_frame = cv2.flip(output_frame, 1)
            text_x_pos = 20; text_y_pos = 40
            if current_status_text:
                text_color = (255, 255, 255) # Default white
                if mode == 'hand': text_color = (0, 255, 0) # Green
                elif mode == 'face': text_color = (255, 255, 0) # Yellow
                elif mode == 'motion': text_color = (0, 255, 255) # Cyan
                elif mode == 'idle': text_color = (200, 200, 200) # Gray
                (tw, th), bl = cv2.getTextSize(current_status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(flipped_display_frame, (text_x_pos-5, text_y_pos-th-bl+5), (text_x_pos+tw+5, text_y_pos+bl+5), (0,0,0), cv2.FILLED)
                cv2.putText(flipped_display_frame, current_status_text, (text_x_pos, text_y_pos+bl//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            mode_txt = f"Mode: {mode.upper()}" if mode != 'idle' else "Mode: IDLE"
            cv2.putText(flipped_display_frame, f"{mode_txt} (1:H 2:F 3:M q:Q)", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("ElderCare Unified System", flipped_display_frame)

            # --- Video Recording ---
            if video_writer and video_writer.isOpened() and perform_processing:
                if flipped_display_frame is not None and flipped_display_frame.size > 0 :
                    try:
                        if video_writer.isOpened():
                            video_writer.write(flipped_display_frame)
                        else:
                            print("WARN: VideoWriter became closed.")
                    except Exception as write_err:
                        print(f"!!! ERROR writing frame: {write_err}")
                        log_event("error", f"Video write failed: {write_err}")
                else:
                    print(f"WARN: Skipped writing invalid frame {frame_count}.")

            # --- Handle Keys ---
            key = cv2.waitKey(1) & 0xFF
            new_mode = None
            if key == ord('1'): new_mode = 'hand'
            elif key == ord('2'): new_mode = 'face'
            elif key == ord('3'): new_mode = 'motion'
            elif key == ord('q'):
                print("Exit key")
                log_event("info", "App stop key")
                break
            if new_mode and new_mode != mode:
                print(f"\nSwitching to {new_mode.capitalize()} Mode")
                log_event("mode_change", f"Mode -> {new_mode}")
                mode = new_mode
                # Reset state variables
                hand_buffer=[];last_hand_spoken='None';last_hand_time=0
                expression_buffer=[];last_expression_spoken='None';last_expression_time=0
                last_motion_spoken='None';last_motion_time=0
                prev_positions={k:None for k in prev_positions}; frame_count=0
                no_hand_start_time = None; played_no_hand_audio = False
                no_face_start_time = None; played_no_face_audio = False
                no_motion_start_time = None; played_no_motion_audio = False
                played_no_person_audio = False

    # --- Cleanup ---
    print("Exiting loop...")
    cap.release()
    cv2.destroyAllWindows()
    if video_writer and video_writer.isOpened():
        video_writer.release(); print(f"Rec stop: {video_filename}"); log_event("info", f"Rec Stop: {video_filename}")
    if conn:
        conn.close(); conn = None; cursor = None; print("DB closed.")
    print("Resources released.")

# --- GUI Function (Unchanged) ---
def show_log_gui():
    global conn, cursor
    if not PYQT5_AVAILABLE:
        print("PyQt5 lib not found.")
        return 1
    if conn:
        print("Closing main DB conn before GUI.")
        try: conn.close()
        except Exception as e: print(f"Err closing conn: {e}"); conn = None; cursor = None
    gui_conn = None
    try:
        gui_conn = sqlite3.connect(DB_PATH); gui_cursor = gui_conn.cursor(); print("GUI DB connected.")
        app = QApplication.instance()
        if app is None: app = QApplication(sys.argv)
        window = QMainWindow(); window.setWindowTitle("ElderCare Log"); window.setGeometry(100, 100, 750, 450)
        table = QTableWidget(window); table.setColumnCount(3); table.setHorizontalHeaderLabels(["Timestamp", "Type", "Content"]); table.setEditTriggers(QTableWidget.NoEditTriggers); table.setAlternatingRowColors(True)
        gui_cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC"); rows = gui_cursor.fetchall(); table.setRowCount(len(rows)); print(f"Logs found: {len(rows)}")
        for i, row in enumerate(rows):
            for j, val in enumerate(row): table.setItem(i, j, QTableWidgetItem(str(val)))
        table.resizeColumnsToContents(); table.horizontalHeader().setStretchLastSection(True); table.setGeometry(10, 10, 730, 430)
        window.show(); print("Launching GUI..."); exit_code = app.exec_(); print(f"GUI closed (code {exit_code})."); return exit_code
    except Exception as gui_err:
        print(f"GUI Err: {gui_err}"); import traceback; traceback.print_exc(); return 1
    finally:
        if gui_conn: gui_conn.close(); print("GUI DB closed.")

# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    exit_status = 0
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--view-log':
            exit_status = show_log_gui()
        else:
            run_main_app()
    except ImportError as imp_err:
        print(f"Import Error: {imp_err}. Check deps."); exit_status = 1
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exit.")
        try: log_event("info", "App interrupted")
        except Exception: pass
        exit_status = 0
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"CRITICAL Err: {e}")
        try: log_event("critical_error", f"Unhandled: {e}")
        except Exception: pass
        exit_status = 1
    finally:
        if conn:
            try: conn.close(); print("DB closed finally.")
            except Exception: pass
        print(f"Exiting script (status {exit_status}).")
        sys.exit(exit_status)