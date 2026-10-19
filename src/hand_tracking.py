
import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.6
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                coordinates = []
                for landmark in hand_landmarks.landmark:
                    coordinates.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                landmarks_list.append(coordinates)

        return frame, landmarks_list

    def close(self):
        self.hands.close()
