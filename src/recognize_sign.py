
import cv2
import joblib
from pathlib import Path

from hand_tracking import HandTracker

# Project root: SS/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sign_model.joblib"

def main():
    model = joblib.load(MODEL_PATH)
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        tracker.close()
        return

    print("SilentBridge live recognition started.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            display, landmarks = tracker.process_frame(frame)

            if landmarks:
                prediction = model.predict([landmarks[0]])[0]
                label = str(prediction)

                cv2.putText(
                    display,
                    f"Predicted sign: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "Show your hand to the camera",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("SilentBridge - Sign Recognition", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
