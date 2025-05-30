# ✅ webapp.py – Flask backend: logs every event + summarize.csv for daily report

from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime, time, timedelta
from base64 import b64decode
import pandas as pd

app = Flask(__name__)

FACES_DIR = "faces"
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

known_faces = []
known_names = []

ATTENDANCE_FILE = "attendance.csv"
SUMMARY_FILE = "summarize.csv"

def load_faces():
    known_faces.clear()
    known_names.clear()
    print("🔄 Loading faces from directory...")
    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith((".jpg", ".png")):
            path = os.path.join(FACES_DIR, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"[❌] Cannot read image: {filename}")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb)
            if enc:
                known_faces.append(enc[0])
                known_names.append(os.path.splitext(filename)[0])
                print(f"[✅] Loaded: {filename}")
            else:
                print(f"[⚠️] No face found in: {filename}")
    print(f"✅ Total loaded: {len(known_faces)} faces.")

load_faces()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/faces/<filename>")
def serve_face(filename):
    return send_from_directory(FACES_DIR, filename)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        data_url = request.form.get("image", "")
        if "," not in data_url:
            print("❌ No image data received.")
            return jsonify(results=[])

        header, encoded = data_url.split(",", 1)
        img_data = b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ Failed to decode image.")
            return jsonify(results=[])

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)

        if not locs:
            print("❌ No face locations detected.")
            return jsonify(results=[])

        encs = face_recognition.face_encodings(rgb, locs)
        print(f"📸 Detected {len(encs)} face(s) from camera.")
        results = []

        for enc in encs:
            distances = face_recognition.face_distance(known_faces, enc)
            if len(distances) == 0:
                results.append("Unknown")
                continue

            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
                now = datetime.now()
                print(f"[🎯] Best match: {name} (distance={distances[best_match_index]:.2f})")

                with open(ATTENDANCE_FILE, "a") as f:
                    f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"[📝] Logged attendance for {name}")

                update_summary()
            else:
                name = "Unknown"
                print(f"[❌] Match too weak (min distance={distances[best_match_index]:.2f})")
            results.append(name)

        return jsonify(results=results)
    except Exception as e:
        print(f"🔥 Exception in /upload: {e}")
        return jsonify(results=[])

def update_summary():
    try:
        df = pd.read_csv(ATTENDANCE_FILE, names=["name", "datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date

        today = datetime.now().date()
        today_df = df[df["date"] == today]

        summary = []
        start_time = datetime.combine(today, time(8, 0, 0))
        end_time = datetime.combine(today, time(17, 0, 0))

        for name in today_df["name"].unique():
            times = today_df[today_df["name"] == name]["datetime"].sort_values()
            earliest = times.iloc[0]
            latest = times.iloc[-1]

            if earliest > start_time and latest < end_time:
                worked = latest - earliest - timedelta(hours=1)
            elif earliest > start_time and latest >= end_time:
                worked = end_time - earliest - timedelta(hours=1)
            elif earliest <= start_time and latest < end_time:
                worked = latest - start_time - timedelta(hours=1)
            else:
                worked = end_time - start_time - timedelta(hours=1)

            if worked.total_seconds() < 0:
                worked_str = "00:00:00"
            else:
                hours, remainder = divmod(worked.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                worked_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            summary.append({
                "Name": name,
                "Check-in": earliest.strftime("%Y-%m-%d %H:%M"),
                "Check-out": latest.strftime("%Y-%m-%d %H:%M"),
                "Working-time": worked_str
            })

        pd.DataFrame(summary).to_csv(SUMMARY_FILE, index=False)
        print("✅ Summary updated")
    except Exception as e:
        print(f"🔥 Failed to update summary: {e}")

@app.route("/register", methods=["POST"])
def register():
    try:
        name = request.form.get("name", "").strip()
        data_url = request.form.get("image", "")
        if not name or "," not in data_url:
            return jsonify(success=False, message="Missing name or image")

        header, encoded = data_url.split(",", 1)
        img_data = b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(success=False, message="Image decode failed")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        if not encs:
            return jsonify(success=False, message="No face detected in registration image")

        save_path = os.path.join(FACES_DIR, f"{name}.jpg")
        cv2.imwrite(save_path, img)
        load_faces()
        return jsonify(success=True, message=f"Registered {name}")
    except Exception as e:
        print(f"🔥 Exception in /register: {e}")
        return jsonify(success=False, message="Error during registration")

if __name__ == "__main__":
    app.run(debug=True)
