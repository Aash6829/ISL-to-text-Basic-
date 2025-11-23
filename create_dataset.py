import os
import pickle
import cv2
import mediapipe as mp

# ===== Mediapipe setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ===== Paths =====
DATA_DIR = './data'
OUTPUT_FILE = 'data.pickle'

data, labels = [], []

# ===== Iterate through gesture folders =====
for gesture in os.listdir(DATA_DIR):
    gesture_path = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(gesture_path):
        continue

    count = 0
    for img_name in os.listdir(gesture_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(gesture_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_ = [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                data.append(data_aux)
                labels.append(gesture)
                count += 1

    print(f"✔ Processed {count} samples for gesture '{gesture}'")

# ===== Save dataset =====
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Saved {len(data)} samples to '{OUTPUT_FILE}'")
