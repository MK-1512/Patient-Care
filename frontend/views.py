from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.conf import settings
import os
import datetime # Needed for datetime.timezone.utc
from django.utils import timezone # Django's timezone functions
# import pytz # Only needed if doing complex timezone logic beyond settings.TIME_ZONE
import sqlite3
import subprocess

# Define path to the activity log database relative to BASE_DIR from settings
ACTIVITY_DB_PATH = os.path.join(settings.BASE_DIR, 'activity_log.db')

# --- View Functions ---

def index(request):
    """Renders the main welcome page."""
    return render(request, 'frontend/index.html')

def run_detection(request):
    """Starts the background detection script via POST request."""
    if request.method == 'POST':
        script_path = os.path.join(settings.BASE_DIR, 'complete2.py')
        print(f"Attempting to run script at: {script_path}")
        if os.path.exists(script_path):
            try:
                subprocess.Popen(['python3', script_path], cwd=settings.BASE_DIR)
                print("Detection script launched.")
            except Exception as e:
                print(f"Error launching detection script: {e}")
        else:
            print(f"Error: Script not found at {script_path}")
    # Always redirect back to index page
    return redirect('index') # Assumes 'index' is the name of your index URL pattern

@login_required # Requires login
def activity_report(request):
    """Displays the activity log. Admin only."""
    # Requires superuser status
    if not request.user.is_superuser:
        return HttpResponseForbidden("Access Denied: You do not have permission to view this page.")

    logs = []; error_message = None; conn_log = None
    if not os.path.exists(ACTIVITY_DB_PATH):
        error_message = f"DB not found: {ACTIVITY_DB_PATH}"
        print(error_message)
    else:
        # Robust connection handling (try read-only first)
        try:
            conn_log = sqlite3.connect(f"file:{ACTIVITY_DB_PATH}?mode=ro", uri=True, check_same_thread=False)
            cursor_log = conn_log.cursor(); print("Connected to activity DB (read-only).")
            try:
                cursor_log.execute("SELECT timestamp, type, content FROM logs ORDER BY timestamp DESC")
                logs = cursor_log.fetchall(); print(f"Activity log query successful, found {len(logs)} logs.")
            except sqlite3.Error as e: error_message = f"DB query error: {e}"; print(error_message)
        except sqlite3.Error as e:
            print(f"Read-only connection failed ({e}), attempting read-write fallback...")
            try:
                conn_log = sqlite3.connect(ACTIVITY_DB_PATH, check_same_thread=False)
                cursor_log = conn_log.cursor(); print("Connected to activity DB (read-write fallback).")
                cursor_log.execute("SELECT timestamp, type, content FROM logs ORDER BY timestamp DESC")
                logs = cursor_log.fetchall(); print(f"Activity log query successful (fallback), found {len(logs)} logs.")
            except sqlite3.Error as e2: error_message = f"DB error on fallback connection: {e2}"; print(error_message)
        finally:
            if conn_log: conn_log.close(); print("Activity DB connection closed.")

    context = { 'logs': logs, 'error_message': error_message }
    return render(request, 'frontend/report.html', context)

# --- Other simple static views ---
def how_it_works(request): return render(request, 'frontend/how_it_works.html')
def contact(request): return render(request, 'frontend/contact.html')
def team(request): return render(request, 'frontend/team.html')
def image_upload_js_view(request): return render(request, 'frontend/upload_image_js.html')


# --- Corrected view_recordings Function ---
@login_required # Requires login
def view_recordings(request):
    """Lists recorded video files, sorted newest first. Admin only."""

    # Requires superuser status
    if not request.user.is_superuser:
        return HttpResponseForbidden("Access Denied: You do not have permission to view recordings.")

    # Ensure MEDIA_ROOT is set correctly in settings.py
    recordings_dir_abs_path = os.path.join(settings.MEDIA_ROOT, "recordings")
    recordings_data = []
    error_message = None
    print(f"--- [View Recordings] Checking path: {recordings_dir_abs_path}")

    if not os.path.isdir(recordings_dir_abs_path):
        error_message = f"Recordings directory not found at: {recordings_dir_abs_path}"
        print(f"--- [View Recordings] Error: {error_message}")
    else:
        print(f"--- [View Recordings] Directory found. Listing contents...")
        try:
            all_files = os.listdir(recordings_dir_abs_path)
            video_filenames = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov')) and not f.startswith('.')]
            sorted_video_filenames = sorted(video_filenames, reverse=True) # Sort newest first
            print(f"--- [View Recordings] Processing {len(sorted_video_filenames)} sorted video files...")

            for filename in sorted_video_filenames:
                file_path = os.path.join(recordings_dir_abs_path, filename)
                try:
                    file_stat = os.stat(file_path)
                    # Ensure MEDIA_URL is set correctly in settings.py (e.g., '/media/')
                    file_url = f"{settings.MEDIA_URL}recordings/{filename}"
                    # Get modification time (UTC timestamp)
                    naive_utc_dt = datetime.datetime.utcfromtimestamp(file_stat.st_mtime)
                    # Make it timezone-aware using standard library UTC object
                    aware_utc_dt = timezone.make_aware(naive_utc_dt, datetime.timezone.utc)
                    # Convert to project's local timezone
                    try:
                        local_dt = timezone.localtime(aware_utc_dt)
                    except Exception as tz_conv_err:
                         print(f"Warn: TZ Conv Err: {tz_conv_err}. Using UTC.")
                         local_dt = aware_utc_dt # Fallback

                    recordings_data.append({
                        'name': filename,
                        'url': file_url,
                        'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                        'modified_time': local_dt
                    })
                except FileNotFoundError: print(f"Warning: File {filename} gone during stat.")
                except Exception as stat_err: print(f"Error getting stats for {filename}: {stat_err}")
        except Exception as list_err:
            error_message = f"Error listing recordings directory: {list_err}"
            print(f"--- [View Recordings] Error: {error_message}")

    print(f"--- [View Recordings] Found {len(recordings_data)} recordings to display.")
    context = {'recordings': recordings_data, 'error_message': error_message, }
    # Ensure the template path is correct
    return render(request, 'frontend/view_recordings.html', context)