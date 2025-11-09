# 🏥 Patient Care System

This is a **comprehensive AI-powered patient monitoring system** that integrates **real-time detection (AI/ML with OpenCV & MediaPipe)** and a **Django-based web interface** for control, monitoring, and activity review.  
The system enables gesture recognition, facial emotion analysis, motion detection, and continuous video recording with event logging.

---

## 🚀 Key Features

### 🤖 Real-time Detection (`complete2.py`)
- **Multi-Modal Detection:** Switch between **hand**, **face**, and **motion** detection modes.  
- **Gesture Recognition:** Uses MediaPipe `gesture_recognizer.task` model to identify gestures and map them to actions (e.g., `'ILoveYou'` → *Call Family*, `'Closed_Fist'` → *Emergency*).  
- **Facial Expression Analysis:** Hybrid approach combining `face_landmarker.task` blendshape scores with geometric landmark data to detect **happy**, **sad**, or **neutral** expressions.  
- **Motion Detection:** Monitors movement (head, hand, body) and triggers a **“no motion detected”** alert after inactivity delay.  

### 🚨 Alerts & Logging
- **Audio Alerts:** Text-to-speech warnings for emergencies or inactivity using `plyer`.  
- **Desktop Notifications:** Real-time pop-up alerts for critical events.  
- **Event Logging:** All detections and errors recorded in an **SQLite (`activity_log.db`)** database.  
- **Video Recording:** Continuous video recording with detection overlays saved in `/media/recordings/` as `.avi` files.

---

## 🌐 Web Interface (Django)
- **Admin Dashboard:** Provides a unified interface for caregivers or administrators.  
- **Start Detection:** A `run_detection` view triggers the Python detection script in the background.  
- **Activity Report:** Displays detection logs from `activity_log.db` in a tabular HTML format (superuser only).  
- **View Recordings:** Allows playback of recorded `.avi` video files directly from the web portal (superuser only).  

---

## 🧰 Technology Stack

### 🧠 Detection Script
| Category | Technology |
|-----------|-------------|
| **Core** | Python 3 |
| **AI/ML** | OpenCV, MediaPipe |
| **Database** | SQLite3 |
| **Alerts** | plyer |
| **GUI Viewer** | PyQt5 (`--view-log` utility) |

### 🌍 Web Interface
| Category | Technology |
|-----------|-------------|
| **Framework** | Django |
| **Database** | SQLite3 (shared with detection script) |

---

## ⚙️ Setup and Installation

### 1️⃣ Install Python Dependencies  
> It’s highly recommended to use a **virtual environment** before installation.

```bash
pip install django opencv-python mediapipe plyer pyqt5
```

### 2️⃣ Navigate to Project Directory  
```bash
cd mk-1512/patient-care/Patient-Care-0c224ef548fc8c4bd272a76d5a9ebc89d2f3ce19
```

### 3️⃣ Set Up Django Database  
> Creates required tables for the admin/superuser system.

```bash
python manage.py migrate
```

### 4️⃣ Create a Superuser  
> Used to log in to the web interface.

```bash
python manage.py createsuperuser
```

### 5️⃣ Run the Django Web Server  
```bash
python manage.py runserver
```

### 6️⃣ Use the System  
1. Open your browser at: `http://127.0.0.1:8000/`  
2. Log in using your **superuser credentials**.  
3. Click **“Start Detection”** to launch the detection script.  
4. Visit `/activity/` to view detection logs and `/recordings/` to see saved video files.

---

## 🧪 Notes & Usage Tips

- Ensure that your **webcam** is properly connected before starting detection.  
- The detection script and Django server can run concurrently.  
- All event logs are automatically stored in `activity_log.db`.  
- For best performance, run on a system with GPU support (optional).  
- Stop the detection script manually from the terminal if needed (`Ctrl + C`).  

---

## 👨‍💻 Author & Contact

**Mukesh Kumar J**  
- Email: mktech1512@gmail.com  
- LinkedIn: https://linkedin.com/in/mk2003  
- GitHub: https://github.com/MK-1512

---
