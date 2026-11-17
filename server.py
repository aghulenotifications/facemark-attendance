import csv
from datetime import datetime
import io
import os
import pickle
import random
import time
from typing import List
import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response
import face_recognition
import numpy as np

app = FastAPI(title="FaceMark Attendance Server")

ENCODINGS_DIR = "known_encodings"
os.makedirs(ENCODINGS_DIR, exist_ok=True)

live_present = set()
challenge_tracker = {}


def load_known_profiles():
    encodings, names = [], []
    for filename in sorted(os.listdir(ENCODINGS_DIR)):
        if filename.endswith(".pkl"):
            filepath = os.path.join(ENCODINGS_DIR, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                arr = np.asarray(data, dtype=np.float64).reshape(-1)
                if arr.shape == (128,):
                    encodings.append(arr)
                    stem, _ = os.path.splitext(filename)
                    names.append(stem)
    return encodings, names


def compute_eye_aspect_ratio(eye_pts):
    pts = np.asarray(eye_pts, dtype=np.float64)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return 0.30 if horiz == 0 else (v1 + v2) / (2.0 * horiz)


def compute_head_yaw_ratio(landmarks):
    chin = np.asarray(landmarks["chin"], dtype=np.float64)
    nose = np.asarray(landmarks["nose_tip"], dtype=np.float64)
    left_jaw = chin[0]
    right_jaw = chin[16]
    nose_center = nose[2]
    dist_left = np.linalg.norm(nose_center - left_jaw)
    dist_right = np.linalg.norm(nose_center - right_jaw)
    total = dist_left + dist_right
    return 0.5 if total == 0 else dist_left / total


def create_random_challenge():
    step1, step2 = random.sample(["BLINK", "TURN_LEFT", "TURN_RIGHT"], 2)
    return {
        "steps": [step1, step2],
        "current_idx": 0,
        "blink_closed": False,
        "start_time": time.time(),
        "verified": False,
    }
@app.get("/profiles")
def get_profiles():
    _, names = load_known_profiles()
    absent = [p for p in names if p not in live_present]
    return {
        "detections": [],
        "present": sorted(list(live_present)),
        "absent": sorted(absent),
    }


@app.post("/register-bulk")
async def register_bulk(files: List[UploadFile] = File(...)):
    saved = []
    skipped = []

    for uploaded_file in files:
        stem, _ = os.path.splitext(uploaded_file.filename or "")
        base_name = stem.strip()
        if not base_name:
            continue

        img_bytes = await uploaded_file.read()
        arr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            skipped.append(f"{uploaded_file.filename} (invalid image)")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)

        if len(encs) == 0:
            skipped.append(f"{uploaded_file.filename} (no face found)")
        elif len(encs) > 1:
            skipped.append(f"{uploaded_file.filename} (multiple faces found)")
        else:
            save_path = os.path.join(ENCODINGS_DIR, f"{base_name}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(encs[0], f)
            saved.append(base_name)

    _, names = load_known_profiles()
    absent = sorted([p for p in names if p not in live_present])
    return {
        "saved": saved,
        "skipped": skipped,
        "present": sorted(list(live_present)),
        "absent": absent,
    }


@app.post("/scan-frame")
async def scan_frame(
    file: UploadFile = File(...),
    tolerance: float = Form(0.55),
    antispoof: bool = Form(False),
    mirror: bool = Form(True),
):
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"detections": [], "present": sorted(list(live_present)), "absent": []}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = rgb.shape

    known_encodings, known_names = load_known_profiles()
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    landmarks_list = (
        face_recognition.face_landmarks(rgb, locations)
        if antispoof
        else [None] * len(locations)
    )

    results = []
    known_matrix = (
        np.asarray(known_encodings, dtype=np.float64).reshape(-1, 128)
        if known_encodings
        else None
    )

    for (top, right, bottom, left), enc, landmarks in zip(
        locations, encodings, landmarks_list
    ):
        norm_box = {
            "top": round(top / img_h, 4),
            "right": round(right / img_w, 4),
            "bottom": round(bottom / img_h, 4),
            "left": round(left / img_w, 4),
        }

        matched = "Unknown"
        if known_matrix is not None:
            distances = face_recognition.face_distance(known_matrix, enc)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= tolerance:
                matched = known_names[best_idx]

        if matched == "Unknown":
            results.append(
                {"name": "Unknown", "status": "UNMATCHED", "color": "#ff3b30", "box": norm_box}
            )
            continue

        if not antispoof:
            live_present.add(matched)
            results.append(
                {"name": matched, "status": "PRESENT", "color": "#00e676", "box": norm_box}
            )
        elif landmarks is not None:
            state = challenge_tracker.setdefault(matched, create_random_challenge())
            if not state["verified"]:
                if time.time() - state["start_time"] > 12.0:
                    state = create_random_challenge()
                    challenge_tracker[matched] = state

                action = state["steps"][state["current_idx"]]
                passed = False
                if action == "BLINK":
                    ear = (
                        compute_eye_aspect_ratio(landmarks["left_eye"])
                        + compute_eye_aspect_ratio(landmarks["right_eye"])
                    ) / 2.0
                    if ear < 0.23:
                        state["blink_closed"] = True
                    elif state["blink_closed"]:
                        passed = True
                        state["blink_closed"] = False
                else:
                    yaw = compute_head_yaw_ratio(landmarks)
                    if action == "TURN_LEFT":
                        passed = (yaw < 0.36) if mirror else (yaw > 0.64)
                    elif action == "TURN_RIGHT":
                        passed = (yaw > 0.64) if mirror else (yaw < 0.36)

                if passed:
                    state["current_idx"] += 1
                    if state["current_idx"] >= 2:
                        state["verified"] = True

            if state["verified"]:
                live_present.add(matched)
                results.append(
                    {
                        "name": matched,
                        "status": "VERIFIED LIVE",
                        "color": "#00e676",
                        "box": norm_box,
                    }
                )
            else:
                step = state["current_idx"] + 1
                prompt = state["steps"][state["current_idx"]].replace("_", " ")
                results.append(
                    {
                        "name": matched,
                        "status": f"Step {step}/2: {prompt}",
                        "color": "#ff9100",
                        "box": norm_box,
                    }
                )

    absent = sorted([p for p in known_names if p not in live_present])
    return {
        "detections": results,
        "present": sorted(list(live_present)),
        "absent": absent,
    }


@app.post("/reset")
def reset_attendance():
    live_present.clear()
    challenge_tracker.clear()
    _, names = load_known_profiles()
    return {"detections": [], "present": [], "absent": sorted(names)}


@app.get("/download-csv")
def download_csv():
    _, names = load_known_profiles()
    scan_date = datetime.now().strftime("%Y-%m-%d")
    scan_time = datetime.now().strftime("%H:%M:%S")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Profile_ID", "Status", "Date", "Time"])
    for person in names:
        status = "Present" if person in live_present else "Absent"
        writer.writerow([person, status, scan_date, scan_time])
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"},
    )
@app.get("/", response_class=HTMLResponse)
def mobile_client_ui():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>FaceMark Mobile Scanner</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #121212; color: #fff; }
  h2 { margin: 0 0 8px 0; font-size: 19px; }
  .cam-wrap { position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #000; }
  video { width: 100%; display: block; }
  canvas#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .controls { display: flex; gap: 8px; margin: 10px 0; }
  button { flex: 1; padding: 12px 6px; font-size: 14px; font-weight: bold; border: none; border-radius: 8px; cursor: pointer; }
  .start { background: #00e676; color: #000; }
  .stop { background: #ff5252; color: #fff; }
  .flip { background: #448aff; color: #fff; }
  .csv { background: #ffd740; color: #000; }
  .upload-btn { background: #b388ff; color: #000; width: 100%; margin-top: 8px; }
  .status-box { background: #1e1e1e; padding: 10px; border-radius: 8px; margin-bottom: 10px; font-size: 15px; }
  .lists { display: flex; gap: 10px; margin-bottom: 14px; }
  .card { flex: 1; padding: 10px; border-radius: 8px; font-size: 14px; }
  .present { background: #1b5e20; }
  .absent { background: #b71c1c; }
  .reg-box { background: #1e1e1e; padding: 12px; border-radius: 8px; border: 1px solid #333; }
  ul { padding-left: 18px; margin: 6px 0 0 0; }
</style>
</head>
<body>
<h2>FaceMark Mobile Scanner</h2>

<div class="cam-wrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
</div>
<canvas id="capture" style="display:none;"></canvas>

<div class="controls">
  <button id="toggleBtn" class="start" onclick="toggleScan()">Start Camera</button>
  <button class="flip" onclick="switchCamera()">Flip Cam</button>
  <button class="csv" onclick="window.location='/download-csv'">CSV</button>
  <button class="stop" onclick="resetList()">Reset</button>
</div>

<label style="display:block; margin-bottom:10px; font-size:15px;">
  <input type="checkbox" id="antispoof"> Enable Anti-Video-Spoof Challenge
</label>

<div id="status" class="status-box">Tap Start Camera to begin.</div>

<div class="lists">
  <div class="card present"><b>Present (<span id="pCount">0</span>)</b><ul id="pList"></ul></div>
  <div class="card absent"><b>Absent (<span id="aCount">0</span>)</b><ul id="aList"></ul></div>
</div>

<div class="reg-box">
  <b>Add New Profiles (Bulk Photo Upload)</b>
  <div style="font-size:12px; color:#bbb; margin:4px 0 8px 0;">
    Select one or more solo photos. Each file is saved using its filename (e.g., Ashish_Ghule_Roll_25.jpg).
  </div>
  <input type="file" id="profileFiles" accept="image/*" multiple style="width:100%; color:#fff;">
  <button class="upload-btn" onclick="uploadProfiles()">Upload & Create Profiles</button>
  <div id="uploadStatus" style="margin-top:8px; font-size:13px; color:#ffd740;"></div>
</div>
<script>
let stream = null, scanning = false, resetting = false, facingMode = "user";
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const capture = document.getElementById("capture");

async function startCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    video.style.transform = (facingMode === "user") ? "scaleX(-1)" : "none";
    scanning = true;
    document.getElementById("toggleBtn").innerText = "Stop";
    document.getElementById("toggleBtn").className = "stop";
    scanLoop();
  } catch (err) {
    document.getElementById("status").innerText = "Camera blocked: Allow camera permissions in browser.";
  }
}

function toggleScan() {
  if (scanning) {
    scanning = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    clearOverlay();
    document.getElementById("toggleBtn").innerText = "Start Camera";
    document.getElementById("toggleBtn").className = "start";
    document.getElementById("status").innerText = "Camera stopped.";
  } else {
    startCamera();
  }
}

function switchCamera() {
  facingMode = (facingMode === "user") ? "environment" : "user";
  if (scanning) startCamera();
}

function clearOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawBoxes(detections) {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const isMirrored = (facingMode === "user");

  detections.forEach(d => {
    if (!d.box) return;
    const w = (d.box.right - d.box.left) * overlay.width;
    const h = (d.box.bottom - d.box.top) * overlay.height;
    const y = d.box.top * overlay.height;
    const x = isMirrored
      ? (1 - d.box.right) * overlay.width
      : d.box.left * overlay.width;

    ctx.strokeStyle = d.color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    const label = `${d.name} - ${d.status}`;
    ctx.font = "bold 15px -apple-system, sans-serif";
    const textW = ctx.measureText(label).width + 14;
    const bannerY = Math.min(overlay.height - 26, y + h);

    ctx.fillStyle = d.color;
    ctx.fillRect(x, bannerY, Math.max(w, textW), 26);
    ctx.fillStyle = (d.color === "#ff3b30") ? "#fff" : "#000";
    ctx.fillText(label, x + 7, bannerY + 18);
  });
}

async function scanLoop() {
  while (scanning) {
    if (resetting || video.videoWidth === 0) {
      await new Promise(r => setTimeout(r, 150));
      continue;
    }
    const targetW = 360;
    const targetH = Math.round(targetW * (video.videoHeight / video.videoWidth));
    capture.width = targetW;
    capture.height = targetH;
    capture.getContext("2d").drawImage(video, 0, 0, targetW, targetH);

    const blob = await new Promise(r => capture.toBlob(r, "image/jpeg", 0.7));
    if (!blob || !scanning || resetting) continue;

    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    form.append("antispoof", document.getElementById("antispoof").checked);
    form.append("mirror", facingMode === "user");

    try {
      const res = await fetch("/scan-frame", { method: "POST", body: form });
      const data = await res.json();
      if (!resetting && scanning) {
        drawBoxes(data.detections || []);
        updateUI(data);
      }
    } catch (e) {
      await new Promise(r => setTimeout(r, 300));
    }
  }
}

function updateUI(data) {
  const statusEl = document.getElementById("status");
  if (data.detections && data.detections.length > 0) {
    statusEl.innerHTML = data.detections.map(d => `<b>${d.name}</b>: ${d.status}`).join("<br>");
  } else if (scanning) {
    statusEl.innerText = "Scanning for faces...";
  }
  document.getElementById("pCount").innerText = data.present.length;
  document.getElementById("aCount").innerText = data.absent.length;
  document.getElementById("pList").innerHTML = data.present.map(n => `<li>${n}</li>`).join("");
  document.getElementById("aList").innerHTML = data.absent.map(n => `<li>${n}</li>`).join("");
}

async function uploadProfiles() {
  const input = document.getElementById("profileFiles");
  const statusEl = document.getElementById("uploadStatus");
  if (!input.files || input.files.length === 0) {
    statusEl.innerText = "Please select at least one photo first.";
    return;
  }
  statusEl.innerText = `Uploading ${input.files.length} photo(s) and generating encodings...`;
  const form = new FormData();
  for (const file of input.files) {
    form.append("files", file);
  }
  try {
    const res = await fetch("/register-bulk", { method: "POST", body: form });
    const data = await res.json();
    updateUI(data);
    let msg = `Saved ${data.saved.length} profile(s).`;
    if (data.skipped.length > 0) {
      msg += ` Skipped: ${data.skipped.join(", ")}`;
    }
    statusEl.innerText = msg;
    input.value = "";
  } catch (e) {
    statusEl.innerText = "Upload failed. Check server connection.";
  }
}

async function resetList() {
  resetting = true;
  clearOverlay();
  document.getElementById("status").innerText = "Resetting list...";
  const res = await fetch("/reset", { method: "POST" });
  const data = await res.json();
  updateUI(data);
  document.getElementById("status").innerText = "Attendance list cleared.";
  resetting = false;
}

fetch("/profiles").then(r => r.json()).then(updateUI);
</script>
</body>
</html>
"""
