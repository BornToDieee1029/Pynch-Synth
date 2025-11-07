# Hand-Gesture Synth

Pynch Synth is a real-time **AI-vision musical instrument** that turns your hand gestures into MIDI notes.
It uses **MediaPipe Hands** for landmark tracking, **OpenCV** for visuals, and **MIDI (mido + python-rtmidi)** to play any virtual instrument in your DAW.

---

## Features

* **Hand-tracking control**

  * Pinch (ring + thumb) → play a note
  * Vertical movement → pitch / glides
  * Horizontal wiggle → vibrato
* **Two play modes**

  * **Full-range (C3–C6)**
  * **18-key Slice mode** with scrollable window
* **Pitch-bend toggle** (`P`)
* **Transparent modern HUD** with color themes
* **Mouse-scroll piano roll** (like Logic Pro)
* **MIDI output** to GarageBand / Logic / Ableton / FL Studio
* **Optimized tracking**

  * Depth-normalized pinch, smoothing, hysteresis
  * OpenCV multithreading and optimizations

---

## System Requirements

| Component | Requirement                                                 |
| --------- | ----------------------------------------------------------- |
| OS        | macOS / Linux / Raspberry Pi 4 or 5                         |
| Python    | **3.12 .x** (recommended)                                   |
| RAM       | ≥ 2 GB (4 GB + preferred)                                   |
| Camera    | USB / built-in 720p +                                       |
| DAW       | Any MIDI-compatible host (GarageBand, Logic, Ableton, etc.) |

---

## Installation

### 1. Create a virtual environment

```bash
python3.12 -m venv lifeline-venv
source lifeline-venv/bin/activate
pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install numpy==1.26.4
pip install opencv-contrib-python==4.12.0.88
pip install mediapipe==0.10.14
pip install mido python-rtmidi
```

Optional (for direct audio):

```bash
pip install sounddevice
```

---

## Configuration

At the top of the script, enable OpenCV optimizations:

```python
import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
```

Allow **Terminal** to access the **Camera** under
*System Settings → Privacy & Security → Camera*

Enable a virtual MIDI port via macOS **Audio MIDI Setup → IAC Driver**.

---

## Run

```bash
source ~/lifeline-venv/bin/activate
python ~/Downloads/lifeline_instrument.py
```

### Controls

| Key                           | Function                              |
| ----------------------------- | ------------------------------------- |
| **M**                         | Toggle Slice / Full mode              |
| **↑ / ↓ / Shift+↑ / Shift+↓** | Scroll semitone / octave              |
| **Mouse**                     | Drag / wheel to move 18-key window    |
| **P**                         | Toggle pitch-bend                     |
| **V**                         | Toggle overlays                       |
| **C**                         | Cycle color themes                    |
| **L / H / B**                 | Toggle lanes / landmarks / bottom bar |
| **[ / ]**                     | Adjust lane transparency              |
| **Esc**                       | Quit                                  |

---

## Tested Versions

| Library        | Version   |
| -------------- | --------- |
| Python         | 3.12 .x   |
| NumPy          | 1.26.4    |
| OpenCV-contrib | 4.12.0.88 |
| MediaPipe      | 0.10.14   |
| Mido           | latest    |
| python-rtmidi  | latest    |

---

## Notes

* For highest tracking accuracy: bright, even lighting; uncluttered background.
* Raspberry Pi 5 can run the full synth standalone.
* Pico / microcontrollers can act as external MIDI controllers (knobs / scroll / footswitch).

