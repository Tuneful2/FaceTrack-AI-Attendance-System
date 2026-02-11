import os
import subprocess
import time
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
FLASK_APP = os.path.join(BASE_DIR, "src", "server.py")
UI_DIR = os.path.join(BASE_DIR, "ui")

def start_flask():
    print("[INFO] Starting Flask backend...")
    subprocess.Popen(
        [sys.executable, FLASK_APP],
        cwd=BASE_DIR
    )


def start_ui():
    print("[INFO] Starting frontend server...")
    subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        cwd=UI_DIR
    )


if __name__ == "__main__":
    print("\n🚀 Launching Face Recognition Attendance System\n")

    start_flask()
    time.sleep(3)   # allow Flask to start

    start_ui()

    print("\n✅ System is running!")
    print("🌐 Open: http://127.0.0.1:8000/index.html")
    print("❌ Close this window to stop everything")

    input("\nPress ENTER to stop servers...\n")
