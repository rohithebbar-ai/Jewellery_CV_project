# hand_landmarker.py
import cv2
import json
import os
import mediapipe as mp

# Load config
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'config_mediapipe.json')
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)


VIDEO_PATH = cfg['video_path']
LANDMARKS_FILE = cfg['landmarks_file']
RESULTS_DIR = cfg['results_dir']

class HandLandmarker:
    def __init__(self, min_detection_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_video(self) -> None:
        """Run MediaPipe Hands on the input video and save landmarks."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cap = cv2.VideoCapture(VIDEO_PATH)
        frame_idx = 0
        records = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    self.mp_draw.draw_landmarks(frame, hand_landmarks)
                    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    records.append({
                        'frame': frame_idx,
                        'hand': handedness.classification[0].label,
                        'landmarks': coords
                    })

            #cv2.imshow('MediaPipe Hands', frame)
            #if cv2.waitKey(1) & 0xFF == 27:
            #    break
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        with open(LANDMARKS_FILE, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"[INFO] Landmarks saved to {LANDMARKS_FILE}")
        

if __name__ == "__main__":
    # if you’re not using config.json, hard‑code the path here or import your config loader
    hl = HandLandmarker(min_detection_confidence=0.5)
    hl.process_video()
