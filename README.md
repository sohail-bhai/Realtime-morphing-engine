# Realtime Morphing Engine

A real-time computer vision system for face, gesture, deformation, and AR-based visual effects using Python, OpenCV, NumPy, and MediaPipe Tasks.

This project started with a mouth-stretch / rubber-face effect, but the architecture is meant to grow into a broader morphing engine with gesture-controlled effects, face deformation, image/anime morphing, AR filters, avatar control, and recording tools.

---

## Features

- Real-time webcam input
- MediaPipe Tasks-based face tracking
- MediaPipe Tasks-based hand tracking
- Pinch gesture detection
- Gesture-controlled face/mouth deformation
- Anchor locking for stretch/morph control
- Smooth reset after release
- Debug overlay
- Screenshot capture
- Video recording support
- Offline warper test support
- Expandable architecture for future AR effects

---

## Project Structure

```txt
realtime-morphing-engine/
│
├── main.py
├── download_models.py
├── check_environment.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── face_landmarker.task
│   └── hand_landmarker.task
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── trackers.py
│   ├── gesture.py
│   ├── mouth_region.py
│   ├── warper.py
│   ├── blender.py
│   ├── debug_draw.py
│   └── utils.py
│
├── outputs/
│   ├── screenshots/
│   └── recordings/
│
└── assets/
```

---

## Requirements

Recommended:

```txt
Python 3.10, 3.11, or 3.12
Webcam
Windows / Linux / macOS
```

Python 3.13 may not work with every computer vision / ML package yet, so Python 3.12 is the safest option.

---

## Setup

Open terminal or PowerShell inside the project folder.

### 1. Create virtual environment

```powershell
py -m venv .venv
```

### 2. Activate virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

Windows CMD:

```cmd
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Download MediaPipe model files

```powershell
python download_models.py
```

### 6. Check environment

```powershell
python check_environment.py
```

### 7. Run the app

```powershell
python main.py
```

---

## Recommended Run Commands

For smoother performance:

```powershell
python main.py --width 720 --height 405
```

For lower lag:

```powershell
python main.py --width 640 --height 360
```

For stronger morph/stretch:

```powershell
python main.py --strength 1.45
```

If the webcam does not open:

```powershell
python main.py --camera 1
```

---

## Controls

```txt
Q / Esc = Quit
D       = Toggle debug overlay
L       = Toggle landmark display
R       = Reset morph/stretch
S       = Take screenshot
V       = Start/stop recording
+       = Increase strength
-       = Decrease strength
M       = Toggle mirror mode
H       = Toggle help text
```

---

## How It Works

```txt
Webcam Frame
   ↓
Face Landmarker detects face landmarks
   ↓
Hand Landmarker detects hand landmarks
   ↓
Pinch gesture is detected using thumb + index finger distance
   ↓
Nearest mouth/face anchor is selected
   ↓
Anchor locks while dragging
   ↓
Morphing engine warps the region
   ↓
Feather blending smooths the result
   ↓
Final frame is displayed in real time
```

---

## Current Effect

The current version focuses on:

```txt
Gesture-controlled mouth/face deformation
```

The user pinches near the mouth and drags outward. The system locks onto the nearest mouth anchor and morphs the region based on the drag vector.

---

## Planned Features

- Multiple morphing modes
- Eye morphing
- Nose morphing
- Cheek pull effect
- Full face liquify mode
- Anime/image morphing mode
- AR stickers
- Face mask overlays
- Background effects
- Avatar control
- Web version using WebRTC
- GUI control panel
- Effect presets
- Exportable recordings

---

## Troubleshooting

### Camera does not open

Try another camera index:

```powershell
python main.py --camera 1
```

or:

```powershell
python main.py --camera 2
```

### App is laggy

Run at a lower resolution:

```powershell
python main.py --width 640 --height 360
```

### MediaPipe model missing

Run:

```powershell
python download_models.py
```

### Packages not found

Make sure your virtual environment is activated:

```powershell
.\.venv\Scripts\activate
```

Then reinstall:

```powershell
python -m pip install -r requirements.txt
```

### PowerShell blocks activation

Run PowerShell as normal user and use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\activate
```

---

## GitHub Repo Name

Recommended repo name:

```txt
realtime-morphing-engine
```

Recommended display title:

```txt
Realtime Morphing Engine
```

Short name:

```txt
RME
```

---

## License

This project is currently for learning and experimentation. Add a license before public release if you want others to use or contribute to it.
