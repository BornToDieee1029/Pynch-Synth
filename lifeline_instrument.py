import time
import numpy as np
import cv2
import mediapipe as mp
import mido
from mido import Message
import math

# ================== WINDOW ==================
WIN_W, WIN_H = 960, 540
FPS_SMOOTH = 0.9
WINDOW_TITLE = "Lifeline Instrument — Slice & Full Modes"

# ================== MIDI ==================
MIDI_CHANNEL = 0
BEND_RANGE_SEMITONES = 2
MAX_BEND = 8191
DEFAULT_VELOCITY = 100

# ===== Global keyboard range (for both modes) =====
TOTAL_LOW   = 36   # C2
TOTAL_HIGH  = 96   # C7
TOTAL_NOTES = TOTAL_HIGH - TOTAL_LOW + 1

# ===== Full-range mode =====
FULL_LOW  = 48   # C3
FULL_HIGH = 84   # C6
FULL_COUNT = FULL_HIGH - FULL_LOW + 1

# ===== Slice mode (18 keys) =====
SLICE_COUNT = 18
slice_start = 60 - SLICE_COUNT//2   # default window center-ish; clamped at runtime

# Tracking / smoothing
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6
BEND_SMOOTH  = 0.6
RETRIGGER_MIN_GAP = 0.020
CALIB_FRAMES = 30
VIB_SENSITIVITY = 0.35

# Pitch bend enable/disable
enable_bend = True   # toggle with 'P'

# ================== THEMES ==================
THEMES = [
    {   # Ocean
        "lane_colors": ((58, 58, 58), (82, 82, 82)),
        "hilite_fill": (140, 170, 255),
        "hilite_border": (190, 220, 255),
        "cursor": (0, 255, 200),
        "hud_bg": (32, 36, 48),
        "hud_border": (80, 90, 120),
        "bar_text": (230, 230, 230),
        "note_text": (210, 220, 235),
        "glow_color": (150, 200, 255),
        "mini_bg": (28, 28, 32),
        "mini_fill": (120, 160, 255),
        "mini_border": (180, 200, 255)
    },
    {   # Neon
        "lane_colors": ((40, 40, 45), (55, 55, 60)),
        "hilite_fill": (80, 255, 180),
        "hilite_border": (110, 255, 200),
        "cursor": (255, 170, 80),
        "hud_bg": (24, 26, 28),
        "hud_border": (70, 70, 70),
        "bar_text": (240, 240, 240),
        "note_text": (230, 240, 255),
        "glow_color": (120, 255, 200),
        "mini_bg": (24, 24, 26),
        "mini_fill": (120, 255, 200),
        "mini_border": (160, 255, 220)
    },
]
theme_idx = 0

# Transparency
LANE_ALPHA = 0.22
HILITE_ALPHA = 0.33
CURSOR_ALPHA = 0.85

# Toggles
show_lanes = True
show_landmarks = True
show_bottom_bar = True
show_overlays = True
slice_mode = True   # start in 18-key mode

# Mouse / mini-keyboard region (updated each frame)
mini_bounds = (0,0,0,0)  # (x1, y1, x2, y2)
mouse_dragging = False

# ================== MIDI SETUP ==================
names = mido.get_output_names()
if not names:
    print("No MIDI outputs found. Open your DAW or enable a virtual MIDI port, then rerun.")
    raise SystemExit(1)
out_port_name = names[0]
print("MIDI outputs:", names)
print("Using:", out_port_name)
midi_out = mido.open_output(out_port_name)

def send_note_on(note, vel=DEFAULT_VELOCITY):
    midi_out.send(Message('note_on', note=int(np.clip(note, 0, 127)),
                          velocity=int(np.clip(vel, 1, 127)),
                          channel=MIDI_CHANNEL))
def send_note_off(note):
    midi_out.send(Message('note_off', note=int(np.clip(note, 0, 127)),
                          velocity=0, channel=MIDI_CHANNEL))
def send_pitch_bend(norm):  # [-1..+1]
    val = int(np.clip(norm, -1.0, 1.0) * MAX_BEND)
    midi_out.send(Message('pitchwheel', channel=MIDI_CHANNEL, pitch=val))
def midi_note_to_freq(n):
    return 440.0 * (2 ** ((n - 69) / 12.0))

# ================== MEDIA PIPE ==================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ================== HELPERS ==================
NOTE_NAMES_12 = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def note_name(n):
    octave = (n // 12) - 1
    return f"{NOTE_NAMES_12[n%12]}{octave}"

def clamp_slice_start(ss):
    ss = max(TOTAL_LOW, ss)
    ss = min(TOTAL_HIGH - SLICE_COUNT + 1, ss)
    return ss

def y_to_note_full(y_norm):
    note_f = FULL_LOW + (1.0 - y_norm) * (FULL_COUNT - 1)
    base = int(round(np.clip(note_f, FULL_LOW, FULL_HIGH)))
    return base, note_f

def y_to_note_slice(y_norm, ss):
    note_f = ss + (1.0 - y_norm) * (SLICE_COUNT - 1)
    base = int(round(np.clip(note_f, ss, ss + SLICE_COUNT - 1)))
    return base, note_f

def bend_from_semitones(semi):
    semi = np.clip(semi, -BEND_RANGE_SEMITONES, BEND_RANGE_SEMITONES)
    return semi / BEND_RANGE_SEMITONES

def blend_overlay(base_bgr, overlay_bgr, alpha):
    return cv2.addWeighted(overlay_bgr, alpha, base_bgr, 1.0 - alpha, 0)

def draw_gradient_band(img, y1, y2, c1, c2):
    band = np.zeros_like(img)
    h = max(1, y2 - y1)
    for i in range(h):
        t = i / (h-1 + 1e-6)
        col = (
            int(c1[0]*(1-t) + c2[0]*t),
            int(c1[1]*(1-t) + c2[1]*t),
            int(c2[2]*t + c1[2]*(1-t)),
        )
        cv2.line(band, (0, y1+i), (WIN_W, y1+i), col, 1)
    return band

def draw_lanes_modern(frame_bgr, current_lane_idx, theme, t_anim, n_lanes):
    lane1, lane2 = theme["lane_colors"]
    lane_h = WIN_H / n_lanes
    # Semi-transparent gradient lanes
    for i in range(n_lanes):
        y1 = int(i * lane_h)
        y2 = int((i + 1) * lane_h)
        grad = draw_gradient_band(frame_bgr, y1, y2,
                                  lane1 if (i%2==0) else lane2,
                                  lane2 if (i%2==0) else lane1)
        frame_bgr[:] = blend_overlay(frame_bgr, grad, LANE_ALPHA)
    # Animated glow on current lane
    if current_lane_idx is not None:
        y1 = int(current_lane_idx * lane_h)
        y2 = int((current_lane_idx + 1) * lane_h)
        glow = np.zeros_like(frame_bgr)
        pulse = 0.55 + 0.20*math.sin(2*math.pi * t_anim*1.2)
        cv2.rectangle(glow, (0, y1), (WIN_W, y2), theme["glow_color"], -1)
        frame_bgr[:] = blend_overlay(frame_bgr, glow, HILITE_ALPHA * pulse)
        cv2.rectangle(frame_bgr, (0, y1), (WIN_W, y2), theme["hilite_border"], 1)

def draw_note_labels_left(frame_bgr, theme, n_lanes, first_note):
    """Left-side note labels so you see yourself + labels on the left."""
    label_w = 56
    labels = np.zeros_like(frame_bgr)
    cv2.rectangle(labels, (0,0), (label_w, WIN_H), (28,28,30), -1)
    frame_bgr[:] = blend_overlay(frame_bgr, labels, 0.40)
    lane_h = WIN_H / n_lanes
    for i in range(n_lanes):
        y = int(i*lane_h + lane_h*0.62)
        n = first_note + i
        cv2.putText(frame_bgr, note_name(n), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["note_text"], 1, cv2.LINE_AA)

def draw_cursor(frame_bgr, cx, cy, theme):
    cursor = np.zeros_like(frame_bgr)
    cv2.line(cursor, (cx, 0), (cx, WIN_H), (255, 255, 255), 1)
    cv2.circle(cursor, (cx, cy), 10, theme["cursor"], -1)
    frame_bgr[:] = blend_overlay(frame_bgr, cursor, CURSOR_ALPHA)

def draw_top_hud(frame_bgr, mode_text, note, hz, theme, bend_on):
    x, y = 12, 18
    mode_str = f"Mode: {mode_text}   Bend: {'On' if bend_on else 'Off'}"
    cv2.putText(frame_bgr, mode_str, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230,230,235), 1, cv2.LINE_AA)
    txt = "--" if note is None else f"{note_name(note)}  {hz:.1f} Hz"
    cv2.putText(frame_bgr, txt, (x, y+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.76, (245,245,250), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, txt, (x, y+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.76, (20,20,20), 1, cv2.LINE_AA)

def draw_bottom_bar(frame_bgr, txt, theme):
    if not show_bottom_bar:
        return
    cv2.putText(frame_bgr, txt, (10, WIN_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["bar_text"], 1, cv2.LINE_AA)

def draw_mini_keyboard(frame_bgr, theme, ss):
    """Right-side mini keyboard with current 18-key slice rectangle; updates global mini_bounds."""
    global mini_bounds
    mini_w = 18
    x1 = WIN_W - mini_w - 8
    x2 = WIN_W - 8
    y1 = 8
    y2 = WIN_H - 8
    mini_bounds = (x1, y1, x2, y2)

    # background
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), theme["mini_bg"], -1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), theme["mini_border"], 1)

    # map TOTAL range to full height
    h_tot = y2 - y1
    def y_of_note(n):
        f = (n - TOTAL_LOW) / (TOTAL_NOTES - 1)
        return int(y2 - f * h_tot)

    y_top = y_of_note(ss + SLICE_COUNT - 1)
    y_bot = y_of_note(ss)
    cv2.rectangle(frame_bgr, (x1+1, y_top), (x2-1, y_bot), theme["mini_fill"], -1)
    cv2.rectangle(frame_bgr, (x1+1, y_top), (x2-1, y_bot), theme["mini_border"], 1)

def set_slice_from_y(y):
    """Set slice_start by clicking/dragging on the mini keyboard."""
    global slice_start
    x1, y1, x2, y2 = mini_bounds
    h_tot = y2 - y1
    y_clamped = max(y1, min(y2, y))
    # invert: bottom = TOTAL_LOW, top = TOTAL_HIGH
    f = (y2 - y_clamped) / max(1, (h_tot))
    center_note = int(round(TOTAL_LOW + f * (TOTAL_NOTES - 1)))
    slice_start = clamp_slice_start(center_note - SLICE_COUNT // 2)

def on_mouse(event, x, y, flags, userdata):
    """Mouse controls: click/drag in mini keyboard to reposition slice; wheel to scroll."""
    global mouse_dragging, slice_start
    x1, y1, x2, y2 = mini_bounds
    inside = (x1 <= x <= x2) and (y1 <= y <= y2)
    if event == cv2.EVENT_LBUTTONDOWN and inside:
        mouse_dragging = True
        set_slice_from_y(y)
    elif event == cv2.EVENT_MOUSEMOVE and mouse_dragging:
        set_slice_from_y(y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        # flags > 0 usually means wheel up, < 0 wheel down
        delta = +1 if flags > 0 else -1
        slice_start = clamp_slice_start(slice_start + delta)

def main():
    global theme_idx, LANE_ALPHA, show_lanes, show_landmarks, show_bottom_bar, show_overlays, slice_mode, slice_start, enable_bend

    slice_start = clamp_slice_start(slice_start)
    theme = THEMES[theme_idx]

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera found.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    active_note = None
    last_bend = 0.0
    last_pitch_target = 0.0
    last_note_time = 0.0
    was_pinched = False
    pinch_baseline = None
    pinch_thresh = None
    frame_count = 0
    fps = 30.0
    t0 = time.time()

    THUMB_TIP = 4
    RING_TIP  = 16

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIN_W, WIN_H))
        theme = THEMES[theme_idx]
        tick = time.time() - t0

        # Hand tracking (always running)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        current_lane_idx = None
        pinch_info = ""
        cx, cy = None, None

        n_lanes = SLICE_COUNT if slice_mode else FULL_COUNT
        first_note = slice_start if slice_mode else FULL_LOW

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0].landmark
            def p(i): return np.array([hand[i].x, hand[i].y])
            thumb = p(THUMB_TIP); ring = p(RING_TIP)
            pinch_dist = float(np.linalg.norm(thumb - ring))
            pinch_pt   = (thumb + ring) / 2.0
            x_norm, y_norm = float(pinch_pt[0]), float(pinch_pt[1])

            frame_count += 1
            if frame_count <= CALIB_FRAMES:
                pinch_baseline = pinch_dist if pinch_baseline is None else min(pinch_baseline, pinch_dist)
                pinch_thresh = (pinch_baseline * 1.6) if pinch_baseline else 0.05
            if pinch_thresh is None: pinch_thresh = 0.05

            is_pinched = pinch_dist < pinch_thresh
            pinch_info = f"pinch:{pinch_dist:.3f} thr:{pinch_thresh:.3f}"

            if slice_mode:
                base_note, note_exact = y_to_note_slice(y_norm, slice_start)
                current_lane_idx = base_note - slice_start
            else:
                base_note, note_exact = y_to_note_full(y_norm)
                current_lane_idx = base_note - FULL_LOW

            # Smooth target for bends
            last_pitch_target = BEND_SMOOTH * last_pitch_target + (1 - BEND_SMOOTH) * note_exact

            now = time.time()
            x_dev = (x_norm - 0.5)
            vib_semi = x_dev * VIB_SENSITIVITY if enable_bend else 0.0
            cont_semi = (last_pitch_target - base_note + vib_semi) if enable_bend else 0.0
            bend = bend_from_semitones(cont_semi) if enable_bend else 0.0

            # ===== MIDI logic =====
            if is_pinched and not was_pinched:
                send_note_on(base_note, DEFAULT_VELOCITY)
                active_note = base_note
                last_note_time = now
                last_bend = 0.0
                send_pitch_bend(0.0)  # always center on note-on

            if is_pinched and active_note is not None:
                if enable_bend:
                    if abs(bend - last_bend) > 0.0025:
                        send_pitch_bend(bend)
                        last_bend = bend
                # Optional re-center when crossing into a new lane
                if base_note != active_note and (now - last_note_time) > RETRIGGER_MIN_GAP:
                    send_note_off(active_note)
                    send_note_on(base_note, DEFAULT_VELOCITY)
                    active_note = base_note
                    last_note_time = now
                    last_bend = 0.0
                    send_pitch_bend(0.0)

            if (not is_pinched) and was_pinched:
                send_pitch_bend(0.0); last_bend = 0.0
                if active_note is not None:
                    send_note_off(active_note)
                    active_note = None
            was_pinched = is_pinched

            if show_landmarks and show_overlays:
                mp_draw.draw_landmarks(
                    frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

            cx, cy = int(x_norm * WIN_W), int(y_norm * WIN_H)

        else:
            if was_pinched:
                send_pitch_bend(0.0)
                if active_note is not None:
                    send_note_off(active_note)
                    active_note = None
            was_pinched = False

        # ====== VISUALS ======
        if show_overlays:
            if show_lanes:
                draw_lanes_modern(frame, current_lane_idx, theme, t_anim=tick, n_lanes=n_lanes)
                # LEFT-SIDE labels (always left, so you see yourself + labels)
                draw_note_labels_left(frame, theme, n_lanes, first_note)

            if cx is not None and cy is not None:
                draw_cursor(frame, cx, cy, theme)

            # Top HUD (shows Bend On/Off)
            mode_text = "Slice (18 keys)" if slice_mode else "Full range"
            hz = midi_note_to_freq(active_note) if active_note is not None else 0.0
            draw_top_hud(frame, mode_text, active_note, hz, theme, bend_on=enable_bend)

            # Mini keyboard (right) + mouse control
            if slice_mode:
                draw_mini_keyboard(frame, theme, slice_start)

            tips = "Mouse: drag/wheel on mini-kbd | Arrows=Scroll  Shift+Arrows=Oct  M=Mode  P=Bend  V/L/H/B/C/[/]  ESC=quit"
            draw_bottom_bar(frame, tips + f"   |   {pinch_info}", theme)
        else:
            cv2.putText(frame, "Pure Camera View (press V for HUD)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)

        # Show
        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('v'), ord('V')):
            show_overlays = not show_overlays
        elif key in (ord('l'), ord('L')):
            show_lanes = not show_lanes
        elif key in (ord('h'), ord('H')):
            show_landmarks = not show_landmarks
        elif key in (ord('b'), ord('B')):
            show_bottom_bar = not show_bottom_bar
        elif key in (ord('c'), ord('C')):
            theme_idx = (theme_idx + 1) % len(THEMES)
        elif key == ord(']'):
            LANE_ALPHA = min(0.6, LANE_ALPHA + 0.04)
        elif key == ord('['):
            LANE_ALPHA = max(0.02, LANE_ALPHA - 0.04)
        elif key in (ord('m'), ord('M')):
            slice_mode = not slice_mode
        elif key in (ord('p'), ord('P')):
            enable_bend = not enable_bend
            # safety: re-center bend immediately
            send_pitch_bend(0.0)
        # Slice scroll (keyboard)
        elif key == 82 or key == ord('w'):  # Up or 'w'
            if slice_mode:
                slice_start = clamp_slice_start(slice_start + 1)
        elif key == 84 or key == ord('s'):  # Down or 's'
            if slice_mode:
                slice_start = clamp_slice_start(slice_start - 1)
        elif key == ord('W'):               # Shift+W (some terminals)
            if slice_mode:
                slice_start = clamp_slice_start(slice_start + 12)
        elif key == ord('S'):
            if slice_mode:
                slice_start = clamp_slice_start(slice_start - 12)

    # Cleanup
    send_pitch_bend(0.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
