#main.py
from scripts.hand_landmarker import HandLandmarker
from scripts.ring_candidates import RingCandidateExtractor

def run_prototype():
    print("[STEP] Extracting hand landmarks ...")
    landmarker = HandLandmarker()
    landmarker.process_video()
    
    print("[STEP] Extracting ring candidates patches ...")
    extractor = RingCandidateExtractor()
    extractor.crop_detect()

if __name__ == '__main__':
    run_prototype()