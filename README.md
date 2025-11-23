This project implements a real-time Indian Sign Language (ISL) gesture recognition system using Machine Learning and Computer Vision. The model is trained on hand landmark datasets extracted with MediaPipe, and uses a Random Forest classifier to accurately identify ISL gestures. The application processes live webcam input via OpenCV, displays predicted gestures on screen, and includes text-to-speech output to assist communication for users with hearing or speech impairments.

Key Features

Real-time ISL gesture detection through webcam

Automated landmark extraction using MediaPipe Hands

Machine Learning classification using Random Forest

High prediction accuracy and fast inference

Text-to-speech support for accessibility

Tech Stack

Python, OpenCV, MediaPipe, NumPy, Scikit-learn, Pyttsx3