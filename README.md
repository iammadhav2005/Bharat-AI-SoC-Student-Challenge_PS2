<h1>Problem Statement 2 : Touchless HCI for Media Control using Hand Gestures (Jetson Nano / Orin Nano)</h1>

A real-time, edge-AI powered gesture-controlled media playback system built on NVIDIA Jetson Orin Nano.

This project enables users to control media playback (Play/Pause, Stop, Volume Up, Volume Down) using predefined hand gestures captured through a USB camera with completely touchless.

<h2>Project Overview:</h2>

This system implements a contactless Human-Computer Interaction (HCI) mechanism that replaces conventional input devices such as keyboards, remotes, and touchscreens.

It leverages:

1)MediaPipe for real-time hand landmark detection.

2)OpenCV for video capture and processing.

3)MPV controlled via IPC (Inter-Process Communication).

4)Edge processing entirely on the Jetson device (no cloud dependency).

The system ensures:

1)Low latency.

2)High stability.

3)Privacy (fully offline execution).

4)Efficient embedded hardware utilization.

<h2>Features:</h2>

1)Real-time hand landmark detection (21 keypoints per hand).

2)Rule-based gesture classification.

3)Temporal smoothing for stable detection.

4)Dynamic action delay to prevent repeated triggers.

5)IPC-based media control integration.

6)Optimized for embedded edge AI deployment.

| Gesture       | Action       |
| ------------- | ------------ |
| ✋ Open Palm   | Play / Pause |
| ✊ Fist        | Stop         |
| ☝ One Finger  | Volume Up    |
| ✌ Two Fingers | Volume Down  |

<h2>System Architecture</h2>
Live video capture via USB camera (640×480 resolution)

Frame processing using OpenCV

Hand landmark detection using MediaPipe

Gesture classification via rule-based logic

Command transmission to MPV via IPC socket

Real-time media control execution

<h2>Optimization Techniques:</h2>

Increased detection confidence thresholds

Frame-based temporal smoothing

Dynamic delay mechanism for gesture stability

Efficient socket communication

CPU-optimized processing pipeline

Optional frame skipping for reduced load

<h2>Hardware Used:</h2>

NVIDIA Jetson Orin Nano

USB Camera

HDMI Display

<h2>Applications:</h2>

Assistive technology

Smart classrooms

Healthcare (sterile environments)

Smart home media control

Foundation for advanced gesture-based AI systems
