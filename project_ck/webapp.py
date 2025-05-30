from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from base64 import b64decode

app = Flask(__name__)

FACES_DIR = "faces"
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

known_faces = []
known_names = []

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

def already_logged_today(name):
    today = datetime.now().date()
    if not os.path.exists("attendance.csv"):
        return False
    with open("attendance.csv", "r") as f:
        for line in f:
            try:
                n, t = line.strip().split(",")
                if n == name and datetime.strptime(t, "%Y-%m-%d %H:%M:%S").date() == today:
                    return True
            except:
                continue
    return False

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
            if distances[best_match_index] < 0.5:  # adjustable threshold
                name = known_names[best_match_index]
                print(f"[🎯] Best match: {name} (distance={distances[best_match_index]:.2f})")
                if not already_logged_today(name):
                    with open("attendance.csv", "a") as f:
                        f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"[📝] Logged attendance for {name}")
                else:
                    print(f"[⏱️] Already logged today: {name}")
            else:
                name = "Unknown"
                print(f"[❌] Match too weak (min distance={distances[best_match_index]:.2f})")
            results.append(name)

        return jsonify(results=results)
    except Exception as e:
        print(f"🔥 Exception in /upload: {e}")
        return jsonify(results=[])

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
        load_faces()  # refresh memory
        return jsonify(success=True, message=f"Registered {name}")
    except Exception as e:
        print(f"🔥 Exception in /register: {e}")
        return jsonify(success=False, message="Error during registration")

if __name__ == "__main__":
    app.run(debug=True)