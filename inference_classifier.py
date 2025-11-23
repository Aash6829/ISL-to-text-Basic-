import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound
import threading
import queue
import tempfile
import os
import time

# -------------------------------
# Load trained model
# -------------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# -------------------------------
# Speech queue + worker
# -------------------------------
speech_queue = queue.Queue()

def speech_worker():
    """Plays detected words asynchronously using gTTS."""
    while True:
        text = speech_queue.get()
        if text == "_EXIT_":
            break
        try:
            temp_path = os.path.join(tempfile.gettempdir(), f"{time.time()}.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(temp_path)
            playsound(temp_path, block=True)
            os.remove(temp_path)
        except Exception as e:
            print(f"[TTS Error] {e}")
        finally:
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

# -------------------------------
# Mediapipe setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# Gesture labels
# -------------------------------
labels_dict = {0: 'Hello', 1: 'Please', 2: 'Thank You', 3: 'Sorry'}

# -------------------------------
# Temporal filter settings
# -------------------------------
prev_label = None
gesture_start_time = None
GESTURE_HOLD_TIME = 2.0
spoken_labels = set()

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_label = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux, x_, y_ = [], [], []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        detected_label = prediction[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, str(detected_label), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # --- Gesture Hold Logic ---
    if detected_label != prev_label:
        gesture_start_time = time.time()
        prev_label = detected_label
    elif detected_label is not None and (time.time() - gesture_start_time) >= GESTURE_HOLD_TIME:
        if detected_label not in spoken_labels:
            speech_queue.put(detected_label)
            spoken_labels.add(detected_label)
        gesture_start_time = time.time() + 9999

    cv2.imshow("ISL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
speech_queue.put("_EXIT_")
speech_queue.join()
