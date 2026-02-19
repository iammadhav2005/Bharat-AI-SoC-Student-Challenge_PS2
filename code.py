import cv2
import mediapipe as mp
import time
import socket
import json
import os
import numpy as np
from collections import deque

# ---------------- MPV IPC ----------------
SOCKET_PATH = "/tmp/mpv-socket"

def mpv_command(command, args=[]):
    if not os.path.exists(SOCKET_PATH):
        return
    try:
        s = socket.socket(socket.AF_UNIX)
        s.connect(SOCKET_PATH)
        msg = json.dumps({"command": [command] + args})
        s.send((msg + "\n").encode())
        s.close()
    except:
        pass

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)
mp_draw = mp.solutions.drawing_utils

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- Performance Variables ----------------
frame_times = deque(maxlen=30)
gesture_history = deque(maxlen=15)

SMOOTHING_FRAMES = 5
fingers_history = []

accuracy = 0.0
latency_ms = 0.0
fps = 0.0
avg_frame_time_ms = 0.0

last_action_time = 0
paused = False
action_text = ""

print("Gesture Control with Metrics Started")

# ---------------- Finger Detection ----------------
def fingers_up(hand_landmarks):
    fingers = []
    fingers.append(abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x) > 0.04)
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)
    return fingers

# ---------------- Main Loop ----------------
while True:
    frame_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- Inference Latency --------
    inference_start = time.perf_counter()
    result = hands.process(rgb)
    inference_end = time.perf_counter()
    latency_ms = (inference_end - inference_start) * 1000

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(handLms)

            fingers_history.append(fingers)
            if len(fingers_history) > SMOOTHING_FRAMES:
                fingers_history.pop(0)

            fingers = [int(np.mean([f[i] for f in fingers_history]) > 0.5) for i in range(5)]
            count = fingers.count(True)

            gesture_history.append(tuple(fingers))

            # -------- Accuracy (Stability Based) --------
            if len(gesture_history) > 0:
                most_common = max(set(gesture_history), key=gesture_history.count)
                stability = gesture_history.count(most_common) / len(gesture_history)
                accuracy = stability * 100

            now = time.time()

            if now - last_action_time > 1:
                if count == 5:
                    paused = not paused
                    mpv_command("set_property", ["pause", paused])
                    action_text = "Play / Pause"

                elif count == 1:
                    mpv_command("add", ["volume", 5])
                    action_text = "Volume Up"

                elif count == 2:
                    mpv_command("add", ["volume", -5])
                    action_text = "Volume Down"

                elif count == 0:
                    mpv_command("stop")
                    action_text = "Stop"

                else:
                    action_text = ""

                last_action_time = now

    # -------- Frame Metrics --------
    frame_end = time.perf_counter()
    total_frame_time = frame_end - frame_start
    frame_times.append(total_frame_time)

    if len(frame_times) > 0:
        avg_frame_time_ms = np.mean(frame_times) * 1000
        fps = 1 / np.mean(frame_times)

    # ---------------- Display Metrics ----------------
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Latency: {latency_ms:.2f} ms", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Avg Frame Time: {avg_frame_time_ms:.2f} ms", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.putText(frame, f"Gesture Recognition Accuracy: {accuracy:.2f}%", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # -------- Display Current Action --------
    if action_text != "":
        cv2.putText(frame, f"Action: {action_text}", (10,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 3)

    cv2.imshow("Gesture Control - Performance Metrics", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
