# OcuTrack - Eye Closure Detection

OcuTrack is a Python application for real-time eye closure detection using **Dlib** and **OpenCV**.  
The program calculates the **Eye Aspect Ratio (EAR)** and alerts the user if eyes remain closed for too long.

## Features

- Real-time eye tracking via webcam.
- Visual overlay of eye landmarks on screen.
- Alerts when eyes are closed too long:
  - On-screen warning
  - Terminal message
- Configurable thresholds:
  - Eye closure threshold
  - Duration threshold for alert

## Requirements

- Python 3.11+
- Python libraries:
  ```bash
  pip install opencv-python dlib scipy
  ```
