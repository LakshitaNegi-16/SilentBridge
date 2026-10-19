
import csv
import time
from pathlib import Path

import cv2

from src.hand_tracking import HandTracker


# Project paths
DATA_DIR = Path("dataset")
CSV_PATH = DATA_DIR / "landmarks.csv"

# Signs to collect in this first version
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LABEL = 100

DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    tracker = HandTracker(max_hands=1)

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["label"] + [
                f"feature_{i}" for i in range(63)
            ]
            writer.writerow(header)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            tracker.close()
            return
        for label in LABELS:
                collected = 0

                print(f"\nGet ready to collect sign: {label}")
                print("Press SPACE to start collecting this sign.")
                print("Press Q to quit.")

                # Wait for the user to be ready.
                while True:
                    success, frame = cap.read()

                    if not success:
                        print("Could not read from webcam.")
                        return

                    frame = cv2.flip(frame, 1)
                    display = frame.copy()

                    cv2.putText(
                        display,
                        f"Sign: {label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display,
                        "SPACE: start | Q: quit",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow("SilentBridge Data Collection", display)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord(" "):
                        break
                    if key == ord("q"):
                        return

                # Brief pause so you can position your hand.
                start_time = time.time()

                while time.time() - start_time < 2:
                    success, frame = cap.read()
                    if not success:
                        continue

                    frame = cv2.flip(frame, 1)
                    cv2.putText(
                        frame,
                        "Get ready...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("SilentBridge Data Collection", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                # Collect landmark samples for this label.
                while collected < SAMPLES_PER_LABEL:
                    success, frame = cap.read()

                    if not success:
                        print("Could not read from webcam.")
                        break

                    frame = cv2.flip(frame, 1)
                    annotated_frame, landmarks = tracker.process_frame(frame)

                    if landmarks:
                        writer.writerow([label] + landmarks[0])
                        file.flush()
                        collected += 1

                    cv2.putText(
                        annotated_frame,
                        f"Sign: {label}  Samples: {collected}/{SAMPLES_PER_LABEL}",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        "Hold the same sign steadily | Q: quit",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow(
                        "SilentBridge Data Collection",
                        annotated_frame,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Collection stopped by user.")
                        return

                print(f"Collected {collected} samples for {label}.")

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"\nData saved to: {CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cv2.destroyAllWindows()
