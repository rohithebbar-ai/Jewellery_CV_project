import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands


def get_hand_landmarks(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands_detector:
        results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return []
        return results.multi_hand_landmarks
