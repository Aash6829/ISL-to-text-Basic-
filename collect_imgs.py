import os
import cv2
import time

# ===== Configuration =====
DATA_DIR = './data'
GESTURES = ["Hello", "Please", "ThankYou", "Sorry"]
IMAGES_PER_CLASS = 100

# ===== Create main data folder =====
os.makedirs(DATA_DIR, exist_ok=True)

# ===== Open the webcam =====
cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    gesture_path = os.path.join(DATA_DIR, gesture)
    os.makedirs(gesture_path, exist_ok=True)

    print(f"\nCollecting data for '{gesture}'")

    # ---- Ready screen ----
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Could not read frame from camera.")
            break
        cv2.putText(frame, f"Show '{gesture}' & press Q to start",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # ---- Countdown before capture ----
    for sec in [3, 2, 1]:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, str(sec), (250, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 5,
                    (0, 0, 255), 8, cv2.LINE_AA)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1000)

    # ---- Capture images ----
    counter = 0
    while counter < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(gesture_path, f"{counter}.jpg"), frame)
        counter += 1

print("\n✅ Data collection finished.")
cap.release()
cv2.destroyAllWindows()
