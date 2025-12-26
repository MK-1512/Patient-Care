

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

try:
    print("--- OpenCV Build Information ---")
    print(f"OpenCV Version: {cv2.__version__}")
    print("------------------------------")
except Exception as e:
    print(f"Could not get OpenCV build info: {e}")

from mediapipe.framework.formats import landmark_pb2
try:
    from mediapipe.tasks import python as mp_python_task
    from mediapipe.tasks.python import vision as mp_vision_task
    from mediapipe import ImageFormat
    TASKS_AVAILABLE = True
except ImportError:
    print("Warning: mediapipe.tasks not found. Gesture and new Face recognition disabled.")
    TASKS_AVAILABLE = False; mp_python_task = None; mp_vision_task = None; ImageFormat = None
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    print("Warning: 'plyer' library not found. Desktop notifications disabled.")
    PLYER_AVAILABLE = False; notification = None
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem
    PYQT5_AVAILABLE = True
except ImportError:
    print("Warning: PyQt5 not found. GUI log viewer disabled.")
    PYQT5_AVAILABLE = False

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
    if not text or text in ['None', 'Unrecognized Gesture']: return
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

def draw_landmarks_on_image(bgr_image, hands_result=None, pose_result=None, face_result=None):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh_module = mp.solutions.face_mesh
    mp_face_connections = mp_face_mesh_module.FACEMESH_TESSELATION

    if hands_result and hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    if pose_result and pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            bgr_image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

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


def check_geometric_sad(landmarks, w, h):
    """
    Checks for sadness based on vertical distance between mouth corners and lip centers.
    Returns True if sad geometry detected, False otherwise.
    Uses landmark indices relevant to FaceLandmarker (similar to old FaceMesh).
    """
    try:
        if landmarks and len(landmarks) > 291:
            l = landmarks[61]
            r = landmarks[291]
            t = landmarks[13]
            b = landmarks[14]

            if all([l, r, t, b]):
                diff = ((l.y + r.y) / 2 - (t.y + b.y) / 2) * h
                if diff > 4.5:
                    return True
        return False
    except IndexError:
        return False
    except Exception as e:
        print(f"Geometric Sad Check Err: {e}")
        return False


def get_expression_from_blendshapes(blendshapes):
    """
    Analyzes blendshapes from FaceLandmarker result.
    Returns 'happy', 'sad_blendshape', or 'neutral'.
    DEBUG PRINT ENABLED.
    """
    if not blendshapes or len(blendshapes) == 0:
        return "neutral"

    categories = blendshapes[0]
    blendshape_dict = {category.category_name: category.score for category in categories}

    smile_threshold = 0.4
    frown_threshold = 0.20
    brow_down_threshold = 0.15

    avg_smile = (blendshape_dict.get('mouthSmileLeft', 0) + blendshape_dict.get('mouthSmileRight', 0)) / 2
    avg_frown = (blendshape_dict.get('mouthFrownLeft', 0) + blendshape_dict.get('mouthFrownRight', 0)) / 2
    avg_brow_down = (blendshape_dict.get('browDownLeft', 0) + blendshape_dict.get('browDownRight', 0)) / 2

    print(f"Smile: {avg_smile:.2f}, Frown: {avg_frown:.2f}, BrowDown: {avg_brow_down:.2f}")

    if avg_smile > smile_threshold and avg_smile > avg_frown and avg_smile > avg_brow_down:
        return "happy"
    elif (avg_frown > frown_threshold or avg_brow_down > brow_down_threshold) and avg_smile < (smile_threshold * 0.8):
        return "sad_blendshape"
    else:
        return "neutral"


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

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    gesture_recognizer = None
    face_landmarker = None
    if TASKS_AVAILABLE:
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


    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        default_gestures = ['Closed_Fist', 'Thumb_Down', 'Open_Palm', 'Victory', 'Pointing_Up', 'Thumb_Up', 'ILoveYou', 'None']
        assigned_gestures = ['Emergency', 'Not Good', 'Request Doctor', 'All set', 'Request Food', 'Well & Good', 'Call Family', 'Unrecognized Gesture']


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

        def get_landmark_pos(landmarks_list, index):
            if landmarks_list and index < len(landmarks_list):
                lm = landmarks_list[index]
                if lm and hasattr(lm, 'x') and hasattr(lm, 'y') and np.isfinite(lm.x) and np.isfinite(lm.y):
                    return np.array([lm.x * frame_width, lm.y * frame_height])
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERR: Cam frame fail")
                time.sleep(0.1)
                break

            output_frame = frame.copy()
            current_status_text = ""
            perform_processing = (mode != 'idle')

            hands_result, pose_result, face_result = None, None, None

            if perform_processing:
                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = None
                if TASKS_AVAILABLE and ImageFormat:
                    try:
                        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                    except Exception as img_conv_err:
                        print(f"Error creating mp.Image: {img_conv_err}")
                        log_event("error", f"mp.Image creation failed: {img_conv_err}")
                        perform_processing = False

                rgb_frame.flags.writeable = False
                try:
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

                if mode == 'hand':
                    status_set = False
                    hands_on_frame = hands_result and hands_result.multi_hand_landmarks
                    gesture = 'Unrecognized Gesture'

                    if not hands_on_frame:
                        current_status_text = "No hand detected"
                        gesture = 'Unrecognized Gesture'
                        if last_hand_spoken != "No hand detected":
                            last_hand_spoken = "No hand detected"; hand_buffer.clear()
                        if no_hand_start_time is None:
                            no_hand_start_time = current_time
                        elif (current_time - no_hand_start_time) > AUDIO_ALERT_DELAY and not played_no_hand_audio:
                            speak("no hand detected"); log_event("audio_alert", "no hand detected (3s)"); played_no_hand_audio = True
                        hand_buffer.append(gesture)
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
                            if gesture_recognition_result.gestures and gesture_recognition_result.gestures[0]:
                                name = gesture_recognition_result.gestures[0][0].category_name
                                if name in default_gestures:
                                    try:
                                        idx = default_gestures.index(name)
                                        gesture = assigned_gestures[idx]
                                    except (ValueError, IndexError):
                                        gesture = "Unknown Category Error"
                                elif name and name != 'None':
                                    gesture = f"Unknown ({name})"

                            hand_buffer.append(gesture)
                            if len(hand_buffer)>5: hand_buffer.pop(0)
                            stable = max(set(hand_buffer), key=hand_buffer.count) if hand_buffer else 'Unrecognized Gesture'

                            meaningful_gestures = [g for g in assigned_gestures if g != 'Unrecognized Gesture']
                            if stable in meaningful_gestures and (stable != last_hand_spoken or (current_time - last_hand_time) > 3):
                                print(f"Gesture: {stable}"); speak(stable); log_event("gesture", stable)
                                if PLYER_AVAILABLE:
                                    try:
                                        notification.notify(title='Gesture', message=f"Detected: {stable}", timeout=10)
                                    except Exception as e:
                                        print(f"Notify Err:{e}")
                                last_hand_spoken = stable; last_hand_time = current_time
                            elif stable != last_hand_spoken:
                                 last_hand_spoken = stable

                            current_status_text = f"Gesture: {stable}"
                            status_set = True
                            output_frame = draw_landmarks_on_image(output_frame, hands_result=hands_result)
                        except Exception as gesture_err:
                            print(f"Gesture Err:{gesture_err}")
                            log_event("error", f"Gesture process fail: {gesture_err}")
                            current_status_text = "Gesture Error"; status_set = True
                    if not status_set:
                        current_status_text="Gesture: Init..."
                        no_hand_start_time = None; played_no_hand_audio = False


                elif mode == 'face':
                    expr = "Detecting..."
                    current_frame_expression = "neutral"

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
                            else:
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


            flipped_display_frame = cv2.flip(output_frame, 1)
            text_x_pos = 20; text_y_pos = 40
            if current_status_text:
                text_color = (255, 255, 255)
                if mode == 'hand': text_color = (0, 255, 0)
                elif mode == 'face': text_color = (255, 255, 0)
                elif mode == 'motion': text_color = (0, 255, 255)
                elif mode == 'idle': text_color = (200, 200, 200)
                (tw, th), bl = cv2.getTextSize(current_status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(flipped_display_frame, (text_x_pos-5, text_y_pos-th-bl+5), (text_x_pos+tw+5, text_y_pos+bl+5), (0,0,0), cv2.FILLED)
                cv2.putText(flipped_display_frame, current_status_text, (text_x_pos, text_y_pos+bl//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            mode_txt = f"Mode: {mode.upper()}" if mode != 'idle' else "Mode: IDLE"
            cv2.putText(flipped_display_frame, f"{mode_txt} (1:H 2:F 3:M q:Q)", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("ElderCare Unified System", flipped_display_frame)

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
                hand_buffer=[];last_hand_spoken='None';last_hand_time=0
                expression_buffer=[];last_expression_spoken='None';last_expression_time=0
                last_motion_spoken='None';last_motion_time=0
                prev_positions={k:None for k in prev_positions}; frame_count=0
                no_hand_start_time = None; played_no_hand_audio = False
                no_face_start_time = None; played_no_face_audio = False
                no_motion_start_time = None; played_no_motion_audio = False
                played_no_person_audio = False

    print("Exiting loop...")
    cap.release()
    cv2.destroyAllWindows()
    if video_writer and video_writer.isOpened():
        video_writer.release(); print(f"Rec stop: {video_filename}"); log_event("info", f"Rec Stop: {video_filename}")
    if conn:
        conn.close(); conn = None; cursor = None; print("DB closed.")
    print("Resources released.")

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