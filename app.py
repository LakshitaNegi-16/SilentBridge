
import cv2
import mediapipe as mp
import customtkinter as ctk
import joblib
import numpy as np

from pathlib import Path
from PIL import Image, ImageTk


class SilentBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SilentBridge AI")
        self.root.geometry("1000x700")

        # Load the trained model
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "models" / "sign_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}"
            )

        self.model = joblib.load(model_path)
        print("Model loaded:", model_path)

        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title_label = ctk.CTkLabel(
            root,
            text="SilentBridge AI",
            font=("Arial", 28, "bold")
        )
        self.title_label.pack(pady=12)

        self.video_label = ctk.CTkLabel(root, text="")
        self.video_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            root,
            text="Show your hand to the camera",
            font=("Arial", 16)
        )
        self.status_label.pack(pady=8)

        self.prediction_label = ctk.CTkLabel(
            root,
            text="Recognized sign: —",
            font=("Arial", 22, "bold")
        )
        self.prediction_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        success, frame = self.cap.read()

        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Extract x and y coordinates for 21 landmarks.
                features = []
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y])

                # 21 landmarks x 2 coordinates = 42 features
                sample = np.array(features).reshape(1, -1)

                try:
                    prediction = self.model.predict(sample)[0]

                    self.status_label.configure(
                        text="Hand detected — prediction active"
                    )
                    self.prediction_label.configure(
                        text=f"Recognized sign: {prediction}"
                    )

                except Exception as error:
                    self.status_label.configure(
                        text="Prediction error — check terminal"
                    )
                    print("Prediction error:", repr(error))

            else:
                self.status_label.configure(
                    text="Show your hand to the camera"
                )
                self.prediction_label.configure(
                    text="Recognized sign: —"
                )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image.thumbnail((800, 480))

            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        else:
            self.status_label.configure(
                text="Cannot access camera. Check camera permissions."
            )

        self.root.after(15, self.update_frame)

    def close(self):
        self.running = False

        if self.cap.isOpened():
            self.cap.release()

        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = SilentBridgeApp(root)
    root.mainloop()

import cv2
import mediapipe as mp
import customtkinter as ctk
import joblib
import numpy as np

from pathlib import Path
from PIL import Image, ImageTk


class SilentBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SilentBridge AI")
        self.root.geometry("1000x700")

        # Load the trained model
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "models" / "sign_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}"
            )

        self.model = joblib.load(model_path)
        print("Model loaded:", model_path)

        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title_label = ctk.CTkLabel(
            root,
            text="SilentBridge AI",
            font=("Arial", 28, "bold")
        )
        self.title_label.pack(pady=12)

        self.video_label = ctk.CTkLabel(root, text="")
        self.video_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            root,
            text="Show your hand to the camera",
            font=("Arial", 16)
        )
        self.status_label.pack(pady=8)

        self.prediction_label = ctk.CTkLabel(
            root,
            text="Recognized sign: —",
            font=("Arial", 22, "bold")
        )
        self.prediction_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        success, frame = self.cap.read()

        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Extract x and y coordinates for 21 landmarks.
                features = []
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y])

                # 21 landmarks x 2 coordinates = 42 features
                sample = np.array(features).reshape(1, -1)

                try:
                    prediction = self.model.predict(sample)[0]

                    self.status_label.configure(
                        text="Hand detected — prediction active"
                    )
                    self.prediction_label.configure(
                        text=f"Recognized sign: {prediction}"
                    )

                except Exception as error:
                    self.status_label.configure(
                        text="Prediction error — check terminal"
                    )
                    print("Prediction error:", repr(error))

            else:
                self.status_label.configure(
                    text="Show your hand to the camera"
                )
                self.prediction_label.configure(
                    text="Recognized sign: —"
                )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image.thumbnail((800, 480))

            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        else:
            self.status_label.configure(
                text="Cannot access camera. Check camera permissions."
            )

        self.root.after(15, self.update_frame)

    def close(self):
        self.running = False

        if self.cap.isOpened():
            self.cap.release()

        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = SilentBridgeApp(root)
    root.mainloop()
