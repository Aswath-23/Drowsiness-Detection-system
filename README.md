# Simple Drowsiness Detection System

A lightweight drowsiness detection system that uses your laptop's camera to monitor eye closure and plays an alarm when eyes are closed for more than 5 seconds.

## Features
- Real-time eye detection
- Visual feedback with eye tracking
- System alarm when drowsiness is detected
- Simple setup with minimal dependencies

## Setup Instructions

1. Install the required dependency:
```bash
pip install opencv-python==4.8.0.76
```

2. Run the script:
```bash
python simple_drowsiness_detector.py
```

## Usage
1. The program will open your webcam
2. Green rectangles will appear around detected eyes
3. If your eyes are closed for more than 5 seconds, a system alarm will sound
4. Press 'q' to quit the program

## Notes
- Make sure you're in a well-lit environment
- Position yourself so your face is clearly visible to the camera
- The system uses your computer's built-in system alarm sound 